"""
Classes and methods directly related to setting up Parcels simulations and running them.
"""
from datetime import timedelta, datetime
import importlib
import logging
import math
from pathlib import Path
import sys

import numpy as np
from parcels import (
    ParticleSet,
    ErrorCode,
    AdvectionRK4,
    AdvectionRK45,
    ScipyParticle,
    JITParticle,
)

from pyplume import get_logger, utils
from pyplume.dataloaders import load_geo_points
from pyplume.postprocess import ParticleResult
from pyplume.kernels import DeleteParticle, DeleteParticleVerbose


logger = get_logger(__name__)


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
                points.append(
                    [point[0] + i * kwargs["gapsize"], point[1] + j * kwargs["gapsize"]]
                )
        return points
    if pattern["type"] in ("ball", "circle"):
        kwargs = pattern["args"]
        radius = kwargs["radius"]
        npoints = kwargs["numpoints"]
        angs = np.linspace(0, 2 * math.pi, num=npoints)
        points = []
        # don't bother vectorizing
        for ang in angs:
            points.append(
                [radius * np.cos(ang) + point[0], radius * np.sin(ang) + point[1]]
            )
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
    return utils.import_attr(name)


def insert_default_values(self, cfg):
    # TODO implement
    pass


class ParcelsSimulation:
    def __init__(
        self,
        name,
        grid,
        spawn_points=None,
        particle_type=None,
        save_dir=None,
        snapshot_interval=None,
        kernels=None,
        time_range=None,
        repetitions=None,
        repeat_dt=None,
        instances_per_spawn=None,
        simulation_dt=None,
    ):
        self.name = name
        self.grid = grid
        self.time_range = time_range
        self.repetitions = repetitions
        self.snapshot_interval = snapshot_interval
        self.repeat_dt = repeat_dt
        self.instances_per_spawn = instances_per_spawn
        self.simulation_dt = simulation_dt
        self.times, _, _ = grid.get_coords()

        # load spawn points
        if isinstance(spawn_points, (str, dict)):
            lats, lons = load_geo_points(
                **utils.wrap_in_kwarg(spawn_points, key="data")
            )
            spawn_points = np.array([lats, lons]).T
        elif isinstance(spawn_points, (list, np.ndarray)):
            spawn_points = spawn_points
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

        if isinstance(particle_type, str):
            self.particle_type = import_kernel_or_particle(particle_type)
        else:
            self.particle_type = particle_type
        # set up ParticleSet and ParticleFile
        self.pset = ParticleSet(
            fieldset=grid.fieldset,
            pclass=self.particle_type,
            time=time_arr,
            lon=p_lons,
            lat=p_lats,
        )
        # TODO generalize path lol
        self.sim_result_dir = utils.get_dir(
            Path(save_dir)
            / f"simulation_{name}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        self.pfile_path = self.sim_result_dir / f"particlefile.nc"
        self.pfile = self.pset.ParticleFile(self.pfile_path)
        logger.info(
            f"Particle trajectories for {name} will be saved to {self.pfile_path}"
            + f"\n\ttotal particles in simulation: {len(time_arr)}"
        )

        t_start, t_end = self.get_time_bounds(spawn_points)
        self.snap_num = math.floor((t_end - t_start) / snapshot_interval)
        self.last_int = t_end - (self.snap_num * snapshot_interval + t_start)
        if self.last_int == 0:
            # +1 snapshot is from an initial plot
            logger.info(
                "No last interval exists."
                + f"\nNum snapshots to save for {name}: {self.snap_num + 1}"
            )
        else:
            logger.info(f"Num snapshots to save for {name}: {self.snap_num + 2}")

        self.completed = False
        self.parcels_result = None
        self.kernels = []
        for kernel in kernels:
            if isinstance(kernel, str):
                self.kernels.append(import_kernel_or_particle(kernel))
            else:
                self.kernels.append(kernel)
        if len(self.kernels) == 0:
            self.kernels = [AdvectionRK4]
        self.kernel = None
        self.update_kernel()

    def generate_single_particle_spawns(self, **kwargs):
        """Generates spawn information for a single specified location"""
        t_start, t_end = parse_time_range(self.time_range, self.times)
        release = np.datetime64(kwargs.get("release", t_start))
        if release < t_start or release > t_end:
            raise ValueError(
                f"Particle is released {release}, outside of simulation time range {t_start}, {t_end}"
            )
        # convert from datetime to delta seconds
        release = (release - self.times[0]) / np.timedelta64(1, "s")
        t_end = (t_end - self.times[0]) / np.timedelta64(1, "s")
        point = kwargs["point"]
        if len(point) != 2:
            raise ValueError(f"{point} has incorrect point dimensions")
        repetitions = kwargs.get("repetitions", self.repetitions)
        repeat_dt = kwargs.get("repeat_dt", self.repeat_dt)
        instances_per_spawn = kwargs.get(
            "instances_per_spawn", self.instances_per_spawn
        )
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
                return self.time_range[0]
            if isinstance(sp, dict):
                release = sp.get("release", None)
                release = None if release is None else np.datetime64(release)
                if release is not None:
                    if earliest_spawn is None:
                        earliest_spawn = release
                    else:
                        earliest_spawn = (
                            release if release < earliest_spawn else earliest_spawn
                        )
                else:
                    # position that defaults to START
                    return self.time_range[0]
        return earliest_spawn if earliest_spawn is not None else self.time_range[0]

    def get_time_bounds(self, spawns):
        if self.time_range[0] == "START":
            earliest_spawn = self.get_earliest_spawn(spawns)
            self.time_range[0] = earliest_spawn
        t_start, t_end = parse_time_range(self.time_range, self.times)
        if (
            t_start < self.times[0]
            or t_end < self.times[0]
            or t_start > self.times[-1]
            or t_end > self.times[-1]
        ):
            # TODO time extrapolation
            # raise ValueError(
            #     "Start and end times of simulation are out of bounds\n"
            #     + f"ParcelsSimulation range: ({t_start}, {t_end})\n"
            #     + f"Allowed domain: ({self.times[0]}, {self.times[-1]})"
            # )
            pass
        t_start = (t_start - self.times[0]) / np.timedelta64(1, "s")
        t_end = (t_end - self.times[0]) / np.timedelta64(1, "s")
        return t_start, t_end

    def exec_pset(self, runtime):
        self.pset.execute(
            self.kernel,
            runtime=timedelta(seconds=runtime),
            dt=timedelta(seconds=self.simulation_dt),
            recovery={ErrorCode.ErrorOutOfBounds: DeleteParticleVerbose},
            output_file=self.pfile,
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
            logger.info("Particle set is empty, simulation loop not run.")
            return False
        self.pre_loop(iteration, interval)
        self.exec_pset(interval)
        if len(self.pset) == 0:
            logger.info("Particle set empty after execution, no post-loop run.")
            return False
        self.post_loop(iteration, interval)
        return True

    def execute(self):
        if self.completed:
            raise RuntimeError("ParcelsSimulation has already completed.")
        for i in range(self.snap_num):
            if not self.simulation_loop(i, self.snapshot_interval):
                break

        # run the last interval (the remainder) if needed
        if self.last_int != 0:
            self.simulation_loop(self.snap_num, self.last_int)

        # self.pfile.export()
        # ParticleFile exports when it closes
        self.pfile.close()
        self.completed = True
        self.parcels_result = ParticleResult(
            self.pfile_path, sim_result_dir=self.sim_result_dir, snapshot_interval=self.snapshot_interval
        )
        self.parcels_result.add_grid(self.grid)
        return self.parcels_result
