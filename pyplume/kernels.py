"""Create custom particle classes and kernels inside here"""
import math
from operator import attrgetter
import sys

import numpy as np
from parcels import JITParticle, Variable
from parcels import ParcelsRandom


ParcelsRandom.seed(42)


class ThreddsParticle(JITParticle):
    """
    Not actually Thredds specific, just a particle that tracks its own lifetime, spawntime,
    and when it goes out of bounds.
    """
    lifetime = Variable("lifetime", initial=0, dtype=np.float32)
    spawntime = Variable("spawntime", initial=attrgetter("time"), dtype=np.float32)
    # out of bounds
    oob = Variable("oob", initial=0, dtype=np.int32)


def AgeParticle(particle, fieldset, time):
    """Kernel to age particles."""
    particle.lifetime += particle.dt


def RandomWalk5cm(particle, fieldset, time):
    """
    Adds random noise to particle movement (ripped from the plume tracker).

    Adds random noise of 5 cm/s, with mean 0.
    """
    uerr = 5.0 / 100  # 5 cm/s uncertainty with radar
    th = 2 * math.pi * ParcelsRandom.random()  # randomize angle of error
    # convert from degrees to m
    u_conv = 1852.0 * 60 * math.cos(particle.lat * math.pi / 180)  # lon convert
    v_conv = 1852.0 * 60  # lat convert
    u_n = uerr * math.cos(th)
    v_n = uerr * math.sin(th)
    dx = u_n * particle.dt
    dy = v_n * particle.dt
    # undo conversion
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


# NOTE it is impossible to make these kernels configurable
# if you want to use a different time, you need to make the kernel yourself
def DeleteAfter3Days(particle, fieldset, time):
    """Deletes a particle after 3 days"""
    LIFETIME = 259200
    if particle.lifetime > LIFETIME:
        particle.delete()


def DeleteParticle(particle, fieldset, time):
    """Deletes a particle. Mainly for use with the recovery kernel."""
    particle.delete()


def DeleteParticleVerbose(particle, fieldset, time):
    print(f"Particle [{particle.id}] lost "
          f"({particle.time}, {particle.depth}, {particle.lat}, {particle.lon})", file=sys.stderr)
    particle.delete()


def WindModify3Percent(particle, fieldset, time):
    """please dont use this yet idk what im doing"""
    wu = fieldset.WU[time, particle.depth, particle.lat, particle.lon]
    wv = fieldset.WV[time, particle.depth, particle.lat, particle.lon]
    # convert from degrees/s to m/s
    u_conv = 1852 * 60 * math.cos(particle.lat * math.pi / 180)
    v_conv = 1852 * 60
    wu_conv = wu * 0.03 / u_conv
    wv_conv = wv * 0.03 / v_conv
    particle.lon += wu_conv * particle.dt
    particle.lat += wv_conv * particle.dt


def AdvectionRK4BorderCheck(particle, fieldset, time):
    """
    Same Runge-Kutta method, but calculates on the coastline grid if
    the particle is close enough to it.

    Requires the fieldset to have a CUV grid (coastline UV)

    TODO: add special error message that CUV grid is required if it's not there
    """
    (u1, v1) = fieldset.CUV[particle]
    if math.fabs(u1) > 0 or math.fabs(v1) > 0:
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        u2, v2 = fieldset.CUV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        u3, v3 = fieldset.CUV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        u4, v4 = fieldset.CUV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    else:
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
