"""
A collection of methods related to plotting.
"""
import sys

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from parcels import FieldSet, ParticleSet, JITParticle, plotting
import xarray as xr


def get_carree_axis(domain, land=True):
    ext = [domain["W"], domain["E"], domain["S"], domain["N"]]
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    if land:
        ax.add_feature(cartopy.feature.COASTLINE)
    return ax


def get_carree_gl(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels, gl.right_labels = (False, False)
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    return gl


def plot_trajectories(paths, domain, legend=True, scatter=True):
    """
    Takes in Parcels ParticleFile netcdf file paths and creates plots of the
    trajectories on the same plot.

    Args:
        paths (array-like): array of paths to the netcdfs
        domain (dict)
    """
    ax = get_carree_axis(domain)
    gl = get_carree_gl(ax)

    for p in paths:
        p_ds = xr.open_dataset(p)
        # now I'm not entirely sure how matplotlib deals with
        # nan values, so if any show up, damnit
        for i in range(len(p_ds["lat"])):
            name = p.split("/")[-1].split(".")[0]
            if scatter:
                ax.scatter(p_ds["lon"][i], p_ds["lat"][i])
            ax.plot(p_ds["lon"][i], p_ds["lat"][i], label=name)
            # plot starting point as a black X
            ax.plot(p_ds["lon"][i][0], p_ds["lat"][i][0], 'kx')
    if legend:
        ax.legend()
    plt.title("Particle trajectories")
    plt.show()


def plot_particles_ps(fs, lats, lons):
    """
    Quick and dirty way to graph a collection of particles using ParticleSet.show()

    Args:
        fs (parcels.FieldSet)
        lats (array-like): 1-d array of particle latitude values
        lons (array-like): 1-d array of particle longitude values
    """
    if len(lats) == 0 or len(lons) == 0:
        print("Empty lat and lon lists given")
        return
    pset = ParticleSet(fs, pclass=JITParticle, lon=lons, lat=lats)
    pset.show()


def plot_particles(lats, lons, ages, domain, land=True, show=True, savefile=None, part_size=4):
    ax = get_carree_axis(domain, land)
    gl = get_carree_gl(ax)

    if ages is None:
        plt.scatter(lons, lats, s=part_size)
    else:
        plt.scatter(lons, lats, c=ages, edgecolors="k", vmin=0, vmax=vmax, s=part_size)
        plt.colorbar()

    if show:
        plt.show()

    if savefile is not None:
        plt.savefig(savefile)
        print(f"Plot saved to {savefile}.png", file=sys.stderr)
        plt.close()


def plot_particles_age(ps, domain, show_time=None, field=None, land=True, savefile=None, vmax=None, field_vmax=None):
    """
    A scuffed version of ParticleSet.show().
    Colors particles to visualize the particle ages.
    The arguments for this method are essentially the same as ParticleSet.show().

    Args:
        ps (parcels.ParticleSet)
        field_vmax (float): max value for the vector field.
    """
    show_time = ps[0].time if show_time is None else show_time
    ext = [domain["W"], domain["E"], domain["S"], domain["N"]]
    p_size = len(ps)
    lats = np.zeros(p_size)
    lons = np.zeros(p_size)
    ages = np.zeros(p_size)

    for i in range(p_size):
        p = ps[i]
        lats[i] = p.lat
        lons[i] = p.lon
        ages[i] = p.lifetime

    ages /= 86400  # seconds in a day

    if field is None:
        plot_particles(lats, lons, ages, domain, land=land, show=False)
        time_str = plotting.parsetimestr(ps.fieldset.U.grid.time_origin, show_time)
        plt.title(f"Particle ages (days){time_str}")
    else:
        print("Particle age display cannot be used with fields. Showing field only.", file=sys.stderr)
        if field == "vector":
            field = ps.fieldset.UV
        # vector values will always be above 0
        _, fig, ax, _ = plotting.plotfield(field=field, show_time=show_time,
                                           domain=domain, land=land, vmin=0, vmax=field_vmax,
                                           titlestr="Particles and ")
        ax.scatter(lons, lats)

    if savefile is None:
        plt.show()
    else:
        plt.savefig(savefile)
        print(f"Plot saved to {savefile}.png", file=sys.stderr)
        plt.close()
