"""
A collection of methods related to plotting.
"""
from pathlib import Path
import sys

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from parcels import plotting
import xarray as xr

from pyplume.dataloaders import open_dataset


DEFAULT_PARTICLE_SIZE = 4


def get_carree_axis(domain, land=True):
    ext = [domain["W"], domain["E"], domain["S"], domain["N"]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(ext, crs=ccrs.PlateCarree())
    if land:
        ax.add_feature(cartopy.feature.COASTLINE)
    return fig, ax


def get_carree_gl(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.top_labels, gl.right_labels = (False, False)
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
    return gl


def pad_domain(domain, padding):
    domain["S"] -= padding
    domain["N"] += padding
    domain["W"] -= padding
    domain["E"] += padding
    return domain


# TODO finish
def generate_domain(lats, lons, padding=0.005):
    lat_rng = (ds["lat"].values.min(), ds["lat"].values.max())
    if lat_rng[0] < lat_min:
        lat_min = lat_rng[0]
    if lat_rng[1] > lat_max:
        lat_max = lat_rng[1]
    lon_rng = (ds["lon"].values.min(), ds["lon"].values.max())
    if lon_rng[0] < lon_min:
        lon_min = lon_rng[0]
    if lon_rng[1] > lon_max:
        lon_max = lon_rng[1]
    return dict(
        S=lat_min - padding,
        N=lat_max + padding,
        W=lon_min - padding,
        E=lon_max + padding,
    )


def generate_domain_datasets(datasets, padding=0.005):
    """
    Given a list of datasets or paths to particle netcdf files, generate a domain that encompasses
    every position with some padding.
    Will probably break if points go from like 178 to -178 longitude or something.
    """
    lat_min = 90
    lat_max = -90
    lon_min = 180
    lon_max = -180
    for ds in datasets:
        if isinstance(ds, (str, Path)):
            ds = open_dataset(ds)
        lat_rng = (ds["lat"].values.min(), ds["lat"].values.max())
        if lat_rng[0] < lat_min:
            lat_min = lat_rng[0]
        if lat_rng[1] > lat_max:
            lat_max = lat_rng[1]
        lon_rng = (ds["lon"].values.min(), ds["lon"].values.max())
        if lon_rng[0] < lon_min:
            lon_min = lon_rng[0]
        if lon_rng[1] > lon_max:
            lon_max = lon_rng[1]
    return {
        "S": lat_min - padding,
        "N": lat_max + padding,
        "W": lon_min - padding,
        "E": lon_max + padding
    }


def draw_plt(savefile=None, show=False, fit=True, fig=None, figsize=None, verbose=False):
    """
    Args:
        figsize (tuple): (width, height) in inches (or was it the other way around?)
    """
    if fig is None:
        if verbose:
            logging.info("Figure not passed in, figure size unchanged", file=sys.stderr)
    else:
        fig.patch.set_facecolor("w")
        plt.figure(fig.number)
        if figsize is not None:
            fig.set_size_inches(figsize[0], figsize[1])
    plt.draw()
    if show:
        plt.show()
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight" if fit else None)
        if verbose:
            logging.info(f"Plot saved to {savefile}", file=sys.stderr)
        if fig is None:
            plt.close()
        else:
            plt.close(fig)


def draw_trajectories_datasets(datasets, names, domain=None, legend=True, scatter=True, savefile=None, titlestr=None, part_size=DEFAULT_PARTICLE_SIZE, padding=0.0, figsize=None):
    """
    Takes in Parcels ParticleFile datasets or netcdf file paths and creates plots of those
    trajectories on the same plot.

    Args:
        datasets (array-like): array of particle trajectory datasets or paths to nc files containing
         the same type of data
    """
    if len(datasets) != len(names):
        raise ValueError("dataset list length and name list length do not match")
    # automatically generate domain if none is provided
    if domain is None:
        domain = generate_domain_datasets(datasets, padding)
    else:
        pad_domain(domain, padding)
    fig, ax = get_carree_axis(domain)
    gl = get_carree_gl(ax)

    for i, ds in enumerate(datasets):
        if isinstance(ds, (str, Path)):
            ds = open_dataset(ds)
        # now I'm not entirely sure how matplotlib deals with
        # nan values, so if any show up, damnit
        for j in range(ds.dims["traj"]):
            if scatter:
                ax.scatter(ds["lon"][j], ds["lat"][j], s=part_size)
            ax.plot(ds["lon"][j], ds["lat"][j], label=names[i])
            # plot starting point as a black X
            ax.plot(ds["lon"][j][0], ds["lat"][j][0], 'kx')
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    plt.title("Particle trajectories" if titlestr is None else titlestr)

    draw_plt(savefile=savefile, fig=fig, figsize=figsize)


def draw_trajectories(lats, lons, times=None, domain=None, points=True, savefile=None, padding=0.0):
    # TODO finish
    if len(lats.shape) == 1:
        lats = np.array([lats])
    if len(lons.shape) == 1:
        lons = np.array([lons])
    if times is None and len(times.shape) == 1:
        times = np.array([times])
    if domain is None:
        domain = generate_domain(datasets, padding)
    else:
        pad_domain(domain, padding)
    fig, ax = get_carree_axis(domain)
    gl = get_carree_gl(ax)
    for i in range(lats.shape[0]):
        pass


def plot_field(time=None, grid=None, domain=None, land=True, vmax=0.6):
    if grid is None and domain is None:
        raise ValueError("grid or domain must be provided")
    elif grid is not None and domain is None:
        domain = grid.get_domain()
    if grid is None:
        fig, ax = get_carree_axis(domain, land=land)
        get_carree_gl(ax)
    else:
        show_time = None if time is None else int((time - grid.times[0]) / np.timedelta64(1, "s"))
        if show_time is not None and show_time < 0:
            raise ValueError("Particle simulation time domain goes out of bounds")
        _, fig, ax, _ = plotting.plotfield(
            field=grid.fieldset.UV, show_time=show_time, domain=domain, land=land, vmin=0,
            vmax=vmax, titlestr="Particles and "
        )
    return fig, ax

def plot_particles(
    lats, lons, lifetimes=None, time=None, grid=None, domain=None, land=True, vmax=0.6, lifetime_max=None,
    s=20
):
    """
    Plot a collection of particles.

    Args:
        lats
        lons
        ages
        time (np.datetime64)
        grid (SurfaceGrid)
        s: size of the particles

    Returns:
        fig, ax
    """
    if grid is None and domain is None:
        domain = {
            "W": np.nanmin(lons),
            "E": np.nanmax(lons),
            "S": np.nanmin(lats),
            "N": np.nanmax(lats),
        }
        domain = pad_domain(domain, 0.0005)
    elif grid is not None and domain is None:
        domain = grid.get_domain()
    if grid is None:
        fig, ax = get_carree_axis(domain, land=land)
        get_carree_gl(ax)
    else:
        show_time = None if time is None else int((time - grid.times[0]) / np.timedelta64(1, "s"))
        if show_time is not None and show_time < 0:
            raise ValueError("Particle simulation time domain goes out of bounds")
        _, fig, ax, _ = plotting.plotfield(
            field=grid.fieldset.UV, show_time=show_time, domain=domain, land=land, vmin=0,
            vmax=vmax, titlestr="Particles and "
        )
    sc = ax.scatter(lons, lats, c=lifetimes, edgecolor="k", vmin=0, vmax=lifetime_max, s=s)

    if lifetimes is not None:
        cbar_ax = fig.add_axes([0.1, 0, 0.1, 0.1])
        plt.colorbar(sc, cax=cbar_ax)
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.14, posn.y0, 0.04, posn.height])
        cbar_ax.get_yaxis().labelpad = 13
        # super jank label the other colorbar since it's in plotting.plotfield
        cbar_ax.set_ylabel("Age (days)\n\n\n\n\n\nVelocity (m/s)", rotation=270)

    return fig, ax
