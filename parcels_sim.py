from datetime import timedelta
from operator import attrgetter
import os
import math
from pathlib import Path
import sys

import numpy as np
from parcels import ParticleSet, ErrorCode, JITParticle, Variable, AdvectionRK4

import utils
import plot_utils

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
    # print(f"Particle [{particle.id}] lost "
    #       f"({particle.time}, {particle.depth}, {particle.lat}, {particle.lon})", file=sys.stderr)
    particle.delete()


class TimedFrame:
    def __init__(self, time, path):
        self.time = time
        self.path = path


def exec_pset(pset, pfile, runtime, dt):
    k_age = pset.Kernel(AgeParticle)
    k_oob = pset.Kernel(TestOOB)

    pset.execute(
        AdvectionRK4 + k_age + k_oob,
        runtime=timedelta(seconds=runtime),
        dt=timedelta(seconds=dt),
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle},
        output_file=pfile
    )


def save_pset_plot(pset, path, days, domain, field=None):
    plot_utils.plot_particles_age(
        pset, domain, field=field, savefile=path,
        vmax=days, field_vmax=MAX_V
    )


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


def prep_simulation(name, hfrgrid, cfg):
    """
    Note every path returned is a Path object.
    Returns a bunch of objects.
    """
    times, _, _ = hfrgrid.get_coords()
    t_start, t_end = parse_time_range(cfg["time_range"], times)
    t_start = (t_start - times[0]) / np.timedelta64(1, "s")
    t_end = (t_end - times[0]) / np.timedelta64(1, "s")

    if isinstance(cfg["spawn_points"], (str, Path)):
        spawn_points = utils.load_pts_mat(cfg["spawn_points"], "yf", "xf").T
    else:
        spawn_points = np.array(cfg["spawn_points"])

    if cfg["repeat_dt"] <= 0:
        repetitions = 1
    else:
        repetitions = int((t_end - t_start) / cfg["repeat_dt"])
    # the total number of particles that will exist in the simulation
    if cfg["particles_per_dt"] <= 0:
        cfg["particles_per_dt"] = len(spawn_points)
    total = repetitions * cfg["particles_per_dt"]
    time_arr = np.zeros(total)
    for i in range(repetitions):
        start = cfg["particles_per_dt"] * i
        end = cfg["particles_per_dt"] * (i + 1)
        time_arr[start:end] = t_start + cfg["repeat_dt"] * i

    # randomly select spawn points from the given config
    sp_lat = spawn_points.T[0, np.random.randint(0, len(spawn_points), total)]
    sp_lon = spawn_points.T[1, np.random.randint(0, len(spawn_points), total)]
    # vary spawn locations
    p_lats = utils.add_noise(sp_lat, cfg["max_variation"])
    p_lons = utils.add_noise(sp_lon, cfg["max_variation"])

    # set up ParticleSet and ParticleFile
    pset = ParticleSet(
        fieldset=hfrgrid.fieldset, pclass=ThreddsParticle,
        lon=p_lons, lat=p_lats, time=time_arr
    )
    part_path = utils.create_path(utils.PARTICLE_NETCDF_DIR)
    pfile_path = part_path / f"particle_{name}.nc"
    pfile = pset.ParticleFile(pfile_path)
    print(f"Particle trajectories for {name} will be saved to {pfile_path}")
    print(f"    total particles in simulation: {total}")

    snap_num = math.floor((t_end - t_start) / cfg["snapshot_interval"])
    last_int = t_end - (snap_num * cfg["snapshot_interval"] + t_start)
    if last_int == 0:
        print("No last interval exists.")
        print(f"Num snapshots to save for {name}: {snap_num + 2}")
    else:
        print(f"Num snapshots to save for {name}: {snap_num + 3}")
    if snap_num >= 200:
        # TODO move this somewhere else and less hardcoded
        raise Exception(f"Too many snapshots ({snap_num}).")
    snap_path = utils.create_path(utils.PICUTRE_DIR / f"{utils.filename_dict[hfrgrid.resolution]}/{name}")
    print(f"Path to save snapshots to: {snap_path}")

    return pset, pfile, pfile_path, snap_path, snap_num, last_int


def simulation(name, hfrgrid, cfg):
    times, _, _ = hfrgrid.get_coords()
    pset, pfile, pfile_path, snap_path, snap_num, last_int = prep_simulation(name, hfrgrid, cfg)
    if last_int == 0:
        total_iterations = snap_num + 2
    else:
        total_iterations = snap_num + 3
    days = np.timedelta64(times[-1] - times[0], "s") / np.timedelta64(1, "D")
    def save_to(num, zeros=3):
        return str(snap_path / f"snap{str(num).zfill(zeros)}.png")
        # return str(snap_path / f"snap{num}.png")
    def simulation_loop(iteration, interval):
        exec_pset(pset, pfile, interval, cfg["simulation_dt"])
        save_pset_plot(pset, save_to(iteration), days, cfg["shown_domain"], field="vector")
    # save initial plot
    save_pset_plot(pset, save_to(0), days, cfg["shown_domain"], field="vector")
    for i in range(1, snap_num + 1):
        simulation_loop(i, cfg["snapshot_interval"])

    # run the last interval (the remainder) if needed
    if last_int != 0:
        simulation_loop(snap_num + 1, last_int)

    pfile.export()
    pfile.close()

    return pfile_path, snap_path


def generate_sim_gif(name, pic_path, gif_path, gif_delay):
    utils.create_gif(
        gif_delay,
        os.path.join(pic_path, "*.png"),
        gif_path
    )
