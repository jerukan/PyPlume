from datetime import timedelta
from operator import attrgetter
import os
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from parcels import ParticleSet, ErrorCode, ScipyParticle, JITParticle, Variable, AdvectionRK4
from parcels import ParcelsRandom

import utils
from parcels_analysis import ParticleResult
import plot_utils

# ignore annoying deprecation warnings
import warnings
warnings.simplefilter("ignore", UserWarning)
# ignore divide by nan error that happens constantly with parcels
np.seterr(divide='ignore', invalid='ignore')

ParcelsRandom.seed(42)


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


def RandomWalk(particle, fieldset, time):
    cv = 1e-5 * 3600
    uerr = 500
    th = 2 * math.pi * ParcelsRandom.random()
    u_, v_ = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    # convert from degrees/s to m/s
    u_conv = 1852 * 60 * math.cos(particle.lat * math.pi / 180)
    v_conv = 1852 * 60
    u_ *= u_conv
    v_ *= v_conv
    u_n = u_ + uerr * math.cos(th)
    v_n = v_ + uerr * math.sin(th)
    dx = u_n * cv
    dy = v_n * cv
    dx /= u_conv
    dy /= v_conv
    particle.lon += dx
    particle.lat += dy


def TestOOB(particle, fieldset, time):
    """
    Kernel to test if a particle has gone into a location without any ocean current data.
    """
    OOB_THRESH = 1e-14
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    if math.fabs(u) < OOB_THRESH and math.fabs(v) < OOB_THRESH:
        particle.oob = 1
    else:
        particle.oob = 0


def DeleteOOB(particle, fieldset, time):
    """Deletes particles that go out of bounds"""
    OOB_THRESH = 1e-14
    u, v = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    if math.fabs(u) < OOB_THRESH and math.fabs(v) < OOB_THRESH:
        particle.delete()


def DeleteAfterLifetime(particle, fieldset, time):
    LIFETIME = 259200
    if particle.lifetime > LIFETIME:
        particle.delete()


def DeleteParticle(particle, fieldset, time):
    # print(f"Particle [{particle.id}] lost "
    #       f"({particle.time}, {particle.depth}, {particle.lat}, {particle.lon})", file=sys.stderr)
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


class ParcelsSimulation:
    MAX_SNAPSHOTS = 200
    MAX_NUM_LEN = len(str(MAX_SNAPSHOTS))
    MAX_V = 0.6
    PFILE_SAVE_DEFAULT = utils.FILES_ROOT / utils.PARTICLE_NETCDF_DIR

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
        self.pfile_path = utils.create_path(ParcelsSimulation.PFILE_SAVE_DEFAULT) / f"particle_{name}.nc"
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

        self.completed = False
        self.parcels_result = None
        if kernels is None:
            self.kernels = [AgeParticle, DeleteOOB]
        else:
            self.kernels = kernels
        self.kernel = None
        self.update_kernel()

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

    def exec_pset(self, runtime):
        self.pset.execute(
            self.kernel,
            runtime=timedelta(seconds=runtime),
            dt=timedelta(seconds=self.cfg["simulation_dt"]),
            recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
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
        self.parcels_result = ParticleResult(self.pfile_path, cfg=self.cfg)
        self.parcels_result.add_grid(self.hfrgrid)
