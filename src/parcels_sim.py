"""
Classes and methods directly related to setting up Parcels simulations and running them.
"""
from datetime import timedelta
import importlib
import math
import sys

import numpy as np
from parcels import ParticleSet, ErrorCode, AdvectionRK4, AdvectionRK45, ScipyParticle, JITParticle

import src.utils as utils
from src.parcels_analysis import ParticleResult
from src.parcels_kernels import DeleteParticle, DeleteParticleVerbose

# ignore annoying deprecation warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
# ignore divide by nan error that happens constantly with parcels
np.seterr(divide='ignore', invalid='ignore')


def parse_time_range(time_range, time_list):
    """
    Args:
        time_range (array-like): some array with 2 items
         'START' and 'END' are parsed as the start and end of time_list respectively
         an integer represents a delta time in hours
        time_list (array-like): sorted list of timestamps

    Returns:
        np.datetime64, np.datetime64
    """
    if time_range[0] == "START":
        t_start = time_list[0]
    elif isinstance(time_range[0], np.datetime64):
        t_start = time_range[0]
    else:
        try:
            t_start = int(time_range[0])
        except (ValueError, TypeError):
            t_start = np.datetime64(time_range[0])

    if time_range[1] == "END":
        t_end = time_list[-1]
    elif isinstance(time_range[1], np.datetime64):
        t_end = time_range[1]
    else:
        try:
            t_end = int(time_range[1])
        except (ValueError, TypeError):
            t_end = np.datetime64(time_range[1])

    if isinstance(t_start, int) and isinstance(t_end, int):
        raise TypeError("Must have at least one date in the time range")
    if isinstance(t_start, int):
        t_start = t_end - np.timedelta64(t_start)
    if isinstance(t_end, int):
        t_end = t_start + np.timedelta64(t_end)

    return t_start, t_end


def create_with_pattern(point, pattern):
    """Takes single point, returns list of points"""
    if "type" not in pattern:
        return [point]
    if pattern["type"] == "grid":
        kwargs = pattern["args"]
        if kwargs["size"] % 2 != 1 and kwargs["size"] >= 1:
            raise ValueError("Grid size must be a positive odd integer")
        points = []
        radius = kwargs["size"] // 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                points.append([point[0] + i * kwargs["gapsize"], point[1] + j * kwargs["gapsize"]])
        return points
    raise ValueError(f"Unknown pattern {pattern}")    


def import_kernel_or_particle(name):
    """Returns class instances by name"""
    if name == "AdvectionRK4":
        return AdvectionRK4
    if name == "AdvectionRK45":
        return AdvectionRK45
    if name == "ScipyParticle":
        return ScipyParticle
    if name == "JITParticle":
        return JITParticle
    mod = importlib.import_module("src.parcels_kernels")
    try:
        return getattr(mod, name)
    except AttributeError as err:
        raise AttributeError(f"Kernel {name} not found in parcels_kernels.py") from err


def insert_default_values(self, cfg):
    # TODO implement
    pass


class ParcelsSimulation:
    PFILE_SAVE_DEFAULT = utils.FILES_ROOT / utils.PARTICLE_NETCDF_DIR

    def __init__(self, name, hfrgrid, cfg):
        self.name = name
        self.hfrgrid = hfrgrid
        self.cfg = cfg
        self.times, _, _ = hfrgrid.get_coords()

        # load spawn points
        if isinstance(cfg["spawn_points"], (str, dict)):
            lats, lons = utils.load_geo_points(**utils.get_path_cfg(cfg["spawn_points"]))
            spawn_points = np.array([lats, lons]).T
        elif isinstance(cfg["spawn_points"], (list, np.ndarray)):
            spawn_points = cfg["spawn_points"]
        else:
            raise ValueError("Invalid spawn point format in config")

        time_arr = []
        p_lats = []
        p_lons = []
        for spawn_point in spawn_points:
            if isinstance(spawn_point, (list, tuple, np.ndarray)):
                ts, lats, lons = self.generate_single_particle_spawns(point=spawn_point)
            elif isinstance(spawn_point, dict):
                ts, lats, lons = self.generate_single_particle_spawns(**spawn_point)
            else:
                raise TypeError(f"{spawn_point} is an unknown value or type")
            time_arr.extend(ts)
            p_lats.extend(lats)
            p_lons.extend(lons)

        # set up ParticleSet and ParticleFile
        self.pset = ParticleSet(
            fieldset=hfrgrid.fieldset, pclass=import_kernel_or_particle(cfg["particle_type"]),
            time=time_arr, lon=p_lons, lat=p_lats
        )
        if "save_dir_pfile" in cfg and cfg["save_dir_pfile"] not in (None, ""):
            self.pfile_path = utils.create_path(cfg["save_dir_pfile"]) / f"particle_{name}.nc"
        else:
            self.pfile_path = utils.create_path(ParcelsSimulation.PFILE_SAVE_DEFAULT) / f"particle_{name}.nc"
        self.pfile = self.pset.ParticleFile(self.pfile_path)
        print(f"Particle trajectories for {name} will be saved to {self.pfile_path}")
        print(f"    total particles in simulation: {len(time_arr)}")

        t_start, t_end = self.get_time_bounds(spawn_points)
        self.snap_num = math.floor((t_end - t_start) / cfg["snapshot_interval"])
        self.last_int = t_end - (self.snap_num * cfg["snapshot_interval"] + t_start)
        if self.last_int == 0:
            # +1 snapshot is from an initial plot
            print("No last interval exists.")
            print(f"Num snapshots to save for {name}: {self.snap_num + 1}")
        else:
            print(f"Num snapshots to save for {name}: {self.snap_num + 2}")

        self.completed = False
        self.parcels_result = None
        self.kernels = [import_kernel_or_particle(kernel) for kernel in cfg["kernels"]]
        if len(self.kernels) == 0:
            self.kernels = [AdvectionRK4]
        self.kernel = None
        self.update_kernel()

    def generate_single_particle_spawns(self, **kwargs):
        """Generates spawn information for a single specified location"""
        t_start, t_end = parse_time_range(self.cfg["time_range"], self.times)
        release = np.datetime64(kwargs.get("release", t_start))
        if release < t_start:
            raise ValueError(f"Particle is released {release}, before simulation start {t_start}")
        # convert from datetime to delta seconds
        release = (release - self.times[0]) / np.timedelta64(1, "s")
        t_end = (t_end - self.times[0]) / np.timedelta64(1, "s")
        point = kwargs["point"]
        if len(point) != 2:
            raise ValueError(f"{point} has incorrect point dimensions")
        repetitions = kwargs.get("repetitions", self.cfg["repetitions"])
        repeat_dt = kwargs.get("repeat_dt", self.cfg["repeat_dt"])
        instances_per_spawn = kwargs.get("instances_per_spawn", self.cfg["instances_per_spawn"])
        if repetitions is None:
            repetitions = -1
        if repeat_dt is None:
            repeat_dt = -1
        if instances_per_spawn is None:
            instances_per_spawn = 1
        pattern = kwargs.get("pattern", {})
        points = create_with_pattern(point, pattern)
        # calculate number of times particles will be spawned
        if repeat_dt <= 0:
            repetitions = 1
        elif repetitions <= 0:
            repetitions = int((t_end - release) / repeat_dt)
        else:
            if repetitions * repeat_dt >= (t_end - release):
                raise ValueError("Too many repetitions")
        spawn_points = np.tile(points, (instances_per_spawn, 1))
        num_spawns = len(spawn_points)
        # the total number of particles that will exist in the simulation
        total = repetitions * num_spawns
        time_arr = np.zeros(total)
        for i in range(repetitions):
            start = num_spawns * i
            end = num_spawns * (i + 1)
            time_arr[start:end] = release + repeat_dt * i
        p_lats = spawn_points.T[0, np.tile(np.arange(num_spawns), repetitions)]
        p_lons = spawn_points.T[1, np.tile(np.arange(num_spawns), repetitions)]
        return time_arr, p_lats, p_lons

    def add_kernel(self, kernel):
        if kernel in self.kernels:
            raise ValueError(f"{kernel} is already in the list of kernels.")
        self.kernels.append(kernel)
        self.update_kernel()

    def update_kernel(self):
        self.kernel = self.pset.Kernel(self.kernels[0])
        for k in self.kernels[1:]:
            self.kernel += self.pset.Kernel(k)

    def get_earliest_spawn(self, spawns):
        earliest_spawn = None
        for sp in spawns:
            if isinstance(sp, (list, tuple, np.ndarray)):
                # position that defaults to START
                return self.cfg["time_range"][0]
            if isinstance(sp, dict):
                release = sp.get("release", None)
                release = None if release is None else np.datetime64(release)
                if release is not None:
                    if earliest_spawn is None:
                        earliest_spawn = release
                    else:
                        earliest_spawn = release if release < earliest_spawn else earliest_spawn
                else:
                    # position that defaults to START
                    return self.cfg["time_range"][0]
        return earliest_spawn if earliest_spawn is not None else self.cfg["time_range"][0]

    def get_time_bounds(self, spawns):
        if self.cfg["time_range"][0] == "START":
            earliest_spawn = self.get_earliest_spawn(spawns)
            self.cfg["time_range"][0] = earliest_spawn
        t_start, t_end = parse_time_range(
            self.cfg["time_range"], self.times
        )
        if (t_start < self.times[0] or t_end < self.times[0] or
            t_start > self.times[-1] or t_end > self.times[-1]):
            raise ValueError("Start and end times of simulation are out of bounds\n" +
                f"Simulation range: ({t_start}, {t_end})\n" +
                f"Allowed domain: ({self.times[0]}, {self.times[-1]})")
        t_start = (t_start - self.times[0]) / np.timedelta64(1, "s")
        t_end = (t_end - self.times[0]) / np.timedelta64(1, "s")
        return t_start, t_end

    def exec_pset(self, runtime):
        self.pset.execute(
            self.kernel,
            runtime=timedelta(seconds=runtime),
            dt=timedelta(seconds=self.cfg["simulation_dt"]),
            recovery={ErrorCode.ErrorOutOfBounds: DeleteParticleVerbose},
            output_file=self.pfile
        )

    def pre_loop(self, iteration, interval):
        """Can override this hook"""
        pass

    def post_loop(self, iteration, interval):
        """Can override this hook"""
        pass

    def simulation_loop(self, iteration, interval):
        # yes 2 checks are needed to prevent it from breaking
        if len(self.pset) == 0:
            print("Particle set is empty, simulation loop not run.", file=sys.stderr)
            return False
        self.pre_loop(iteration, interval)
        self.exec_pset(interval)
        if len(self.pset) == 0:
            print("Particle set empty after execution, no post-loop run.", file=sys.stderr)
            return False
        self.post_loop(iteration, interval)
        return True

    def execute(self):
        if self.completed:
            raise RuntimeError("Simulation has already completed.")
        for i in range(self.snap_num):
            if not self.simulation_loop(i, self.cfg["snapshot_interval"]):
                break

        # run the last interval (the remainder) if needed
        if self.last_int != 0:
            self.simulation_loop(self.snap_num, self.last_int)

        self.pfile.export()
        self.pfile.close()
        self.completed = True
        self.parcels_result = ParticleResult(self.pfile_path, cfg=self.cfg)
        self.parcels_result.add_grid(self.hfrgrid)