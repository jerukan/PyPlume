from datetime import timedelta
from operator import attrgetter
import os
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from parcels import ParticleSet, ErrorCode, JITParticle, Variable, AdvectionRK4

import utils
import plot_utils

# ignore annoying deprecation warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
# ignore divide by nan error that happens constantly with parcels
np.seterr(divide='ignore', invalid='ignore')

MAX_V = 0.6  # for display purposes only, so the vector field colors don't change every iteration


class ThreddsParticle(JITParticle):
    lifetime = Variable("lifetime", initial=0, dtype=np.float32)
    spawntime = Variable("spawntime", initial=attrgetter("time"), dtype=np.float32)
    # out of bounds
    oob = Variable("oob", initial=0, dtype=np.int32)


def AgeParticle(particle, fieldset, time):
    """
    Kernel to measure particle ages.
    """
    particle.lifetime += particle.dt


def TestOOB(particle, fieldset, time):
    """
    Kernel to test if a particle has gone into a location without any ocean current data.
    """
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    if math.fabs(u) < 1e-14 and math.fabs(v) < 1e-14:
        particle.oob = 1
    else:
        particle.oob = 0


def DeleteParticle(particle, fieldset, time):
    print(f"Particle [{particle.id}] lost "
          f"({particle.time}, {particle.depth}, {particle.lat}, {particle.lon})", file=sys.stderr)
    particle.delete()


def parse_time_range(time_range, time_list):
    """
    Args:
        time_range (array-like): some array with 2 strings
        data (dict)
    """
    if time_range[0] == "START":
        t_start = time_list[0]
    elif isinstance(time_range[0], np.datetime64):
        t_start = time_range[0]
    else:
        try:
            t_start = int(time_range[0])
        except ValueError:
            t_start = np.datetime64(time_range[0])

    if time_range[1] == "END":
        t_end = time_list[-1]
    elif isinstance(time_range[1], np.datetime64):
        t_end = time_range[1]
    else:
        try:
            t_end = int(time_range[1])
        except ValueError:
            t_end = np.datetime64(time_range[1])
            
    if isinstance(t_start, int) and isinstance(t_end, int):
        raise TypeError("Must have at least one date in the time range")
    if isinstance(t_start, int):
        t_start = t_end - np.timedelta64(t_start)
    if isinstance(t_end, int):
        t_end = t_start + np.timedelta64(t_end)
        
    return t_start, t_end


class TimedFrame:
    def __init__(self, time, path):
        self.time = time
        self.path = path

    def __repr__(self):
        return f"([{self.path}] at [{self.time}])"


class ParcelsSimulation:
    MAX_SNAPSHOTS = 200
    MAX_NUM_LEN = len(str(MAX_SNAPSHOTS))
    MAX_V = 0.6
    PFILE_SAVE_DEFAULT = utils.PARTICLE_NETCDF_DIR
    PLOT_SAVE_DEFAULT = utils.PICUTRE_DIR

    def __init__(self, name, hfrgrid, cfg, kernels=None):
        self.name = name
        self.hfrgrid = hfrgrid
        self.cfg = cfg
        self.times, _, _ = hfrgrid.get_coords()
        t_start, t_end = self.get_time_bounds()

        # load spawn points
        try:
            spawn_points = np.array(cfg["spawn_points"], dtype=float)
            if len(spawn_points.shape) != 2 or spawn_points.shape[1] != 2:
                raise ValueError(f"Spawn points is incorrect shape {spawn_points.shape}")
        except ValueError:
            # assume a path was passed in, try to load stuff
            lats, lons = utils.load_pts_mat(cfg["spawn_points"], "yf", "xf")
            spawn_points = np.array([lats, lons]).T

        # calculate number of times particles will be spawned
        if cfg["repeat_dt"] <= 0:
            repetitions = 1
        elif cfg["repetitions"] <= 0:
            repetitions = int((t_end - t_start) / cfg["repeat_dt"])
        else:
            repetitions = cfg["repetitions"]
            if repetitions * cfg["repeat_dt"] >= (t_end - t_start):
                raise ValueError("Too many repetitions")
        num_spawns = len(spawn_points)
        # the total number of particles that will exist in the simulation
        total = repetitions * num_spawns
        time_arr = np.zeros(total)
        for i in range(repetitions):
            start = num_spawns * i
            end = num_spawns * (i + 1)
            time_arr[start:end] = t_start + cfg["repeat_dt"] * i

        p_lats = spawn_points.T[0, np.tile(np.arange(num_spawns), repetitions)]
        p_lons = spawn_points.T[1, np.tile(np.arange(num_spawns), repetitions)]

        # set up ParticleSet and ParticleFile
        self.pset = ParticleSet(
            fieldset=hfrgrid.fieldset, pclass=ThreddsParticle,
            lon=p_lons, lat=p_lats, time=time_arr
        )
        self.pfile_path = utils.create_path(ParcelsSimulation.PFILE_SAVE_DEFAULT / f"particle_{name}.nc")
        self.pfile = self.pset.ParticleFile(self.pfile_path)
        print(f"Particle trajectories for {name} will be saved to {self.pfile_path}")
        print(f"    total particles in simulation: {total}")

        self.snap_num = math.floor((t_end - t_start) / cfg["snapshot_interval"])
        self.last_int = t_end - (self.snap_num * cfg["snapshot_interval"] + t_start)
        if self.last_int == 0:
            # +1 snapshot is from an initial plot
            print("No last interval exists.")
            print(f"Num snapshots to save for {name}: {self.snap_num + 1}")
        else:
            print(f"Num snapshots to save for {name}: {self.snap_num + 2}")
        if self.snap_num >= ParcelsSimulation.MAX_SNAPSHOTS:
            raise Exception(f"Too many snapshots ({self.snap_num}).")
        self.snap_path = utils.create_path(ParcelsSimulation.PLOT_SAVE_DEFAULT / name)
        print(f"Path to save snapshots to: {self.snap_path}")

        if "shown_domain" not in cfg or cfg["shown_domain"] is None:
            self.shown_domain = hfrgrid.get_domain()
        else:
            self.shown_domain = cfg["shown_domain"]
        self.completed = False
        self.lat_pts = []
        self.lon_pts = []
        if kernels is None:
            self.kernels = [AgeParticle, TestOOB]
        else:
            self.kernels = kernels
        self.kernel = None
        self.update_kernel()
        self.days = np.timedelta64(self.times[-1] - self.times[0], "s") / np.timedelta64(1, "D")
        self.plots = []

    def add_line(self, lats, lons):
        self.lat_pts.append(lats)
        self.lon_pts.append(lons)

    def add_kernel(self, kernel):
        if kernel in self.kernels:
            raise ValueError(f"{kernel} is already in the list of kernels.")
        self.kernels.append(kernel)
        self.update_kernel()

    def update_kernel(self):
        self.kernel = AdvectionRK4
        for k in self.kernels:
            self.kernel += self.pset.Kernel(k)

    def get_time_bounds(self):
        t_start, t_end = parse_time_range(self.cfg["time_range"], self.times)
        if (t_start < self.times[0] or t_end < self.times[0] or
            t_start > self.times[-1] or t_end > self.times[-1]):
            raise ValueError("Start and end times of simulation are out of bounds\n" +
                f"Simulation range: ({t_start}, {t_end})\n" +
                f"Allowed domain: ({self.times[0]}, {self.times[-1]})")
        t_start = (t_start - self.times[0]) / np.timedelta64(1, "s")
        t_end = (t_end - self.times[0]) / np.timedelta64(1, "s")
        return t_start, t_end

    def save_pset_plot(self, path):
        part_size = self.cfg.get("part_size", 4)
        fig, ax = plot_utils.plot_particles_age(
            self.pset, self.shown_domain, field="vector", vmax=self.days,
            field_vmax=ParcelsSimulation.MAX_V, part_size=part_size
        )
        for i in range(len(self.lat_pts)):
            ax.scatter(self.lon_pts[i], self.lat_pts[i], s=4)
            ax.plot(self.lon_pts[i], self.lat_pts[i])
        plot_utils.draw_plt(savefile=path, fig=fig)
        self.plots.append(TimedFrame(self.times[0] + np.timedelta64(int(self.pset[0].time), "s"), path))

    def exec_pset(self, runtime):
        self.pset.execute(
            self.kernel,
            runtime=timedelta(seconds=runtime),
            dt=timedelta(seconds=self.cfg["simulation_dt"]),
            recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
            output_file=self.pfile
        )

    def get_plot_save(self, num, zeros=MAX_NUM_LEN):
        """Return path to save a plot to given some number"""
        return str(self.snap_path / f"snap{str(num).zfill(zeros)}.png")

    def pre_loop(self, iteration, interval):
        """Can override this hook"""
        pass

    def post_loop(self, iteration, interval):
        """Can override this hook"""
        pass

    def simulation_loop(self, iteration, interval):
        if len(self.pset) == 0:
            print("Particle set is empty, simulation loop not run.", file=sys.stderr)
            return
        self.pre_loop(iteration, interval)
        self.exec_pset(interval)
        self.save_pset_plot(self.get_plot_save(iteration))
        self.post_loop(iteration, interval)

    def execute(self):
        # clear the folder of pngs (not everything just in case)
        for p in self.snap_path.glob("*.png"):
            p.unlink()
        # save initial plot
        self.simulation_loop(0, 0)
        for i in range(1, self.snap_num + 1):
            self.simulation_loop(i, self.cfg["snapshot_interval"])

        # run the last interval (the remainder) if needed
        if self.last_int != 0:
            self.simulation_loop(self.snap_num + 1, self.last_int)

        self.pfile.export()
        self.pfile.close()
        self.completed = True

    def generate_gif(self, gif_path=None, gif_delay=25):
        if not self.completed:
            raise RuntimeError("Simulation has not been run yet, cannot generate gif")
        if gif_path is None:
            gif_path = ParcelsSimulation.PLOT_SAVE_DEFAULT / f"partsim_{self.name}.gif"
        input_paths = [str(frame.path) for frame in self.plots]
        sp_in = ["magick", "-delay", str(gif_delay)] + input_paths
        sp_in.append(str(gif_path))
        magick_sp = subprocess.Popen(
            sp_in,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = magick_sp.communicate()
        print(f"magick ouptput: {(stdout, stderr)}", file=sys.stderr)
        return gif_path
