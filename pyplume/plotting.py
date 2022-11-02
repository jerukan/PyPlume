"""
A collection of methods related to plotting.
"""
import copy
import datetime
import logging
from pathlib import Path
import sys

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from parcels import plotting
import seaborn as sns
import xarray as xr

from pyplume import get_logger


logger = get_logger(__name__)

DEFAULT_PARTICLE_SIZE = 4


def carree_subplots(shape, projection=None, domain=None, land=False):
    if isinstance(shape, int):
        raise TypeError("Shape as integer not supported by this method. Pass as tuple.")
    if projection is None:
        projection = ccrs.PlateCarree()
    fig = plt.figure()
    axs = np.empty(shape, dtype=matplotlib.axes.Axes)
    i = 1
    for idx, _ in np.ndenumerate(axs):
        _, ax = get_carree_axis(domain=domain, projection=projection, land=land, fig=fig, pos=[*shape, i])
        get_carree_gl(ax)
        axs[idx] = ax
        i += 1
    if axs.shape == (1, 1):
        return fig, axs[0, 0]
    elif axs.shape[0] == 1:
        return fig, axs[0, :]
    elif axs.shape[1] == 1:
        return fig, axs[:, 0]
    return fig, axs


def get_carree_axis(domain=None, projection=None, land=False, fig=None, pos=None):
    """
    Args:
        fig: exiting figure to add to if desired
        pos: position on figure subplots to add axes to
    """
    if projection is None:
        projection = ccrs.PlateCarree()
    if fig is None:
        fig = plt.figure()
    if pos is None:
        pos = 111
    if isinstance(pos, int):
        ax = fig.add_subplot(pos, projection=projection)
    else:
        ax = fig.add_subplot(*pos, projection=projection)
    if domain is not None:
        ext = [domain["W"], domain["E"], domain["S"], domain["N"]]
        ax.set_extent(ext, crs=projection)
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


def generate_domain(lats, lons, padding=0):
    """Will have funky behavior if the coordinate range loops around back to 0."""
    lat_rng = (lats.min(), lats.max())
    lon_rng = (lons.min(), lons.max())
    return dict(
        S=lat_rng[0] - padding,
        N=lat_rng[1] + padding,
        W=lon_rng[0] - padding,
        E=lon_rng[1] + padding,
    )


def generate_domain_datasets(datasets, padding=0):
    """
    Given a list of datasets or paths to particle netcdf files, generate a domain that encompasses
    every position with some padding.
    
    Will have funky behavior if the coordinate range loops around back to 0.
    """
    lat_min = 90
    lat_max = -90
    lon_min = 180
    lon_max = -180
    for ds in datasets:
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


def draw_plt(savefile=None, show=False, fit=True, fig=None, figsize=None):
    """
    Args:
        figsize (tuple): (width, height) in inches (or was it the other way around?)
    """
    if fig is None:
        logger.info("Figure not passed in, figure size unchanged")
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
        logger.info(f"Plot saved to {savefile}")
        if fig is None:
            plt.close()
        else:
            plt.close(fig)


def draw_trajectories_datasets(datasets, names, domain=None, legend=True, scatter=True, savefile=None, titlestr=None, part_size=DEFAULT_PARTICLE_SIZE, padding=0.0, figsize=None):
    """
    Takes in Parcels ParticleFile datasets or netcdf file paths and creates plots of those
    trajectories on the same plot.

    Args:
        datasets (array-like): array of particle trajectory datasets containing the same type of 
         data
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


def plot_vectorfield(
    dataset, show_time=None, domain=None, projection=None, land=True, vmin=None,
    vmax=None, titlestr=None, fig=None, pos=None
):
    if domain is None:
        domain = generate_domain_datasets([dataset])
    if fig is None:
        fig = plt.figure()
    fig, ax = get_carree_axis(domain, projection=projection, land=land, fig=fig, pos=pos)
    get_carree_gl(ax)
    if isinstance(show_time, int):
        idx = show_time
    else:
        if isinstance(show_time, str):
            show_time = np.datetime64(show_time)
        idx = np.where(dataset["time"] == show_time)[0][0] if show_time is not None else 0
    show_time = dataset["time"][idx].values
    U = dataset["U"][idx]
    V = dataset["V"][idx]
    lats = dataset["lat"]
    lons = dataset["lon"]
    spd = U ** 2 + V ** 2
    speed = np.where(spd > 0, np.sqrt(spd), 0)
    vmin = speed.min() if vmin is None else vmin
    vmax = speed.max() if vmax is None else vmax
    ncar_cmap = copy.copy(plt.cm.gist_ncar)
    ncar_cmap.set_over("k")
    ncar_cmap.set_under("w")
    x, y = np.meshgrid(lons, lats)
    u = np.where(speed > 0., U / speed, 0)
    v = np.where(speed > 0., V / speed, 0)
    cs = ax.quiver(
        np.asarray(x), np.asarray(y), np.asarray(u), np.asarray(v), speed, cmap=ncar_cmap,
        clim=[vmin, vmax], scale=50, transform=cartopy.crs.PlateCarree()
    )
    cs.set_clim(vmin, vmax)

    cbar_ax = fig.add_axes([0, 0, 0, 0])
    # fig.subplots_adjust(hspace=0, wspace=0, top=0.925, left=0.1)
    plt.colorbar(cs, cax=cbar_ax)

    def resize_colorbar(event):
        plt.draw()
        posn = ax.get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.04, posn.height])

    fig.canvas.mpl_connect("resize_event", resize_colorbar)
    resize_colorbar(None)

    if titlestr is None:
        titlestr = ""
    else:
        titlestr = f"{titlestr} "
    ax.set_title(f"{titlestr}Velocity field at {show_time.astype('datetime64[s]')}")

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


def draw_particle_density(lats, lons, bins=None, domain=None, ax=None, title="", **kwargs):
    """
    Args:
        kwargs: other arguments to pass into sns.histplot
    """
    if ax is None:
        fig, ax = carree_subplots((1, 1), domain=domain)
    else:
        fig = ax.get_figure()
    bins = bins if bins is not None else 100
    ax.set_title(title)
    sns.histplot(x=lons, y=lats, bins=bins, cbar=True, ax=ax, **kwargs)
    return fig, ax


def draw_coastline(lats, lons, separate_nan=True, domain=None, c=None, ax=None):
    if ax is None:
        fig, ax = carree_subplots((1, 1), domain=domain)
    else:
        fig = ax.get_figure()
    if separate_nan:
        lat_borders = np.split(lats, np.where(np.isnan(lats))[0])
        lon_borders = np.split(lons, np.where(np.isnan(lons))[0])
        for i in range(len(lat_borders)):
            ax.plot(lon_borders[i], lat_borders[i], c=c)
    else:
        ax.plot(lons, lats, c=c)
    return fig, ax
