"""
A collection of methods related to plotting.
"""
import copy
import math

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

from pyplume import get_logger


logger = get_logger(__name__)

DEFAULT_PARTICLE_SIZE = 4


def carree_subplots(shape, projection=None, domain=None, land=False):
    if isinstance(shape, int):
        raise TypeError("Shape as integer not supported by this method. Pass as tuple.")
    if projection is None:
        projection = ccrs.PlateCarree()
    fig = plt.figure()
    axs = np.empty(shape, dtype=mpl.axes.Axes)
    i = 1
    for idx, _ in np.ndenumerate(axs):
        _, ax = get_carree_axis(
            domain=domain, projection=projection, land=land, fig=fig, pos=[*shape, i]
        )
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
    lat_min = math.inf
    lat_max = -math.inf
    lon_min = math.inf
    lon_max = -math.inf
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
        "E": lon_max + padding,
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
        if fit:
            fig.tight_layout()
        plt.savefig(savefile)
        # logger.info(f"Plot saved to {savefile}")
        if fig is None:
            plt.close()
        else:
            plt.close(fig)


def draw_trajectories_datasets(
    datasets,
    names,
    domain=None,
    legend=True,
    scatter=True,
    savefile=None,
    titlestr=None,
    part_size=DEFAULT_PARTICLE_SIZE,
    padding=0.0,
    figsize=None,
):
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
            ax.plot(ds["lon"][j][0], ds["lat"][j][0], "kx")
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05))
    plt.title("Particle trajectories" if titlestr is None else titlestr)

    draw_plt(savefile=savefile, fig=fig, figsize=figsize)


def draw_trajectories(
    lats, lons, times=None, domain=None, points=True, savefile=None, padding=0.0
):
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


def plot_vectorfield(
    dataset,
    show_time=None,
    domain=None,
    projection=None,
    land=False,
    vmin=None,
    vmax=None,
    titlestr=None,
    ax=None,
    color_speed=True,
    cbar=True,
    allow_time_extrapolation=False,
):
    """
    Args:
        show_time: possible values: datetime, int, 'average'
    """
    if domain is None:
        domain = generate_domain_datasets([dataset])
    if ax is None:
        fig, ax = get_carree_axis(domain=domain, projection=projection, land=land)
        get_carree_gl(ax)
    else:
        fig = ax.get_figure()
    dsTime = dataset["time"].values
    dsU = dataset["U"].values
    dsV = dataset["V"].values
    interp = False
    U = None
    V = None
    if show_time is None:
        idx = 0
        show_time = dsTime[0]
    elif isinstance(show_time, str) and show_time.lower() == "average":
        U = np.mean(dsU, axis=0)
        V = np.mean(dsV, axis=0)
    elif isinstance(show_time, int):
        idx = show_time
        # provided index is outside of the time array
        # if negative, will treat as negative index though
        if idx >= len(dsTime):
            if allow_time_extrapolation:
                idx = min(max(0, idx), len(dsTime) - 1)
            else:
                raise ValueError(
                    "Tried plotting vector field oustide of time range. Set allow_time_extrapolation=True"
                )
        show_time = dsTime[idx]
    elif isinstance(show_time, (str, np.datetime64)):
        show_time = np.datetime64(show_time)
        found_idxs = np.where(dsTime == show_time)[0]
        if len(found_idxs) == 0:
            before_idxs = np.where(dsTime <= show_time)[0]
            # provided time is below time range
            if len(before_idxs) == 0:
                if allow_time_extrapolation:
                    idx = 0
                else:
                    raise ValueError(
                        "Tried plotting vector field oustide of time range. Set allow_time_extrapolation=True"
                    )
            else:
                idx = before_idxs[-1]
                interp = True
        else:
            idx = found_idxs[0]
    else:
        raise TypeError(f"show_time of type {type(show_time)} is invalid!")
    if U is None or V is None:
        if interp:
            # provided time is above time range
            if (idx + 1) >= len(dataset["time"]):
                if allow_time_extrapolation:
                    U = dsU[idx].values
                    V = dsV[idx].values
                else:
                    raise ValueError(
                        "Tried plotting vector field oustide of time range. Set allow_time_extrapolation=True"
                    )
            else:
                lower_time = dsTime[idx]
                upper_time = dsTime[idx + 1]
                dist = (show_time - lower_time) / np.timedelta64(1, "s")
                width = (upper_time - lower_time) / np.timedelta64(1, "s")
                ratio = dist / width
                U = (1 - ratio) * dsU[idx] + ratio * dsU[idx + 1]
                V = (1 - ratio) * dsV[idx] + ratio * dsV[idx + 1]
        else:
            U = dsU[idx]
            V = dsV[idx]
    U = np.array(U)
    V = np.array(V)
    lats = dataset["lat"].values
    lons = dataset["lon"].values
    allspd = U ** 2 + V ** 2
    allspeed = np.where(allspd > 0, np.sqrt(allspd), 0)
    vmin = np.nanmin(allspeed) if vmin is None else vmin
    vmax = np.nanmax(allspeed) if vmax is None else vmax
    ncar_cmap = copy.copy(plt.cm.gist_ncar)
    ncar_cmap.set_over("k")
    ncar_cmap.set_under("w")
    x, y = np.meshgrid(lons, lats)
    exists = allspeed > 0.0
    u = np.zeros(U.shape)
    v = np.zeros(V.shape)
    u[exists] = U[exists] / allspeed[exists]
    v[exists] = V[exists] / allspeed[exists]
    if color_speed:
        cs = ax.quiver(
            np.asarray(x),
            np.asarray(y),
            np.asarray(u),
            np.asarray(v),
            allspeed,
            cmap=ncar_cmap,
            clim=[vmin, vmax],
            scale=50,
            transform=cartopy.crs.PlateCarree(),
        )
        cs.set_clim(vmin, vmax)

        if cbar:
            vel_cbar = plt.colorbar(cs)
            vel_cbar.set_label("Current vector velocity (m/s)")
    else:
        cs = ax.quiver(
            np.asarray(x),
            np.asarray(y),
            np.asarray(u),
            np.asarray(v),
            scale=50,
            transform=cartopy.crs.PlateCarree(),
        )

    if titlestr is None:
        titlestr = ""
    elif not titlestr:
        titlestr = False
    else:
        titlestr = f"{titlestr} "
    if titlestr:
        if show_time == "average":
            ax.set_title(f"{titlestr}Velocity field average")
        else:
            ax.set_title(f"{titlestr}Velocity field at {show_time.astype('datetime64[s]')}")

    return fig, ax


def plot_particles(
    lats,
    lons,
    color=None,
    edgecolor=None,
    domain=None,
    land=False,
    vmin=None,
    vmax=None,
    size=None,
    ax=None,
    projection=None,
    cbar=False,
    cbar_label=None,
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
    if ax is None:
        fig, ax = get_carree_axis(domain=domain, projection=projection, land=land)
        get_carree_gl(ax)
    else:
        fig = ax.get_figure()
    sc = ax.scatter(
        lons,
        lats,
        c=color,
        edgecolor=edgecolor,
        vmin=vmin,
        vmax=vmax,
        s=size,
    )
    if cbar:
        cb = plt.colorbar(sc)
        cb.set_label(cbar_label)
    return fig, ax


def plot_particle_density(
    lats, lons, bins=None, domain=None, ax=None, land=True, title="", **kwargs
):
    """
    Args:
        kwargs: other arguments to pass into sns.histplot
    """
    if ax is None:
        fig, ax = carree_subplots((1, 1), land=land, domain=domain)
    else:
        fig = ax.get_figure()
    bins = bins if bins is not None else 100
    ax.set_title(title)
    sns.histplot(
        x=lons,
        y=lats,
        bins=bins,
        cbar=True,
        # stat="probability",
        ax=ax,
        cbar_kws={"label": "Number of particles"},
        **kwargs,
    )
    return fig, ax


def plot_coastline(
    lats, lons, separate_nan=True, domain=None, c=None, linewidth=None, ax=None
):
    if ax is None:
        fig, ax = carree_subplots((1, 1), domain=domain)
    else:
        fig = ax.get_figure()
    if c is None:
        c = "k"
    if separate_nan:
        lat_borders = np.split(lats, np.where(np.isnan(lats))[0])
        lon_borders = np.split(lons, np.where(np.isnan(lons))[0])
        for i in range(len(lat_borders)):
            ax.plot(lon_borders[i], lat_borders[i], c=c, linewidth=linewidth)
    else:
        ax.plot(lons, lats, c=c)
    return fig, ax


def plot_bounding_box(domain, ax, edgecolor="m", linewidth=1, **kwargs):
    width = domain["E"] - domain["W"]
    height = domain["N"] - domain["S"]
    anchor = (domain["W"], domain["S"])
    rect = patches.Rectangle(
        anchor,
        width,
        height,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )
    # Add the patch to the Axes
    ax.add_patch(rect)


def abs_label_map(item):
    if len(item.get_text()) == 0:
        return item
    # matplotlib uses a funny hyphen that doesn't work
    return abs(float(item.get_text().replace("−", "-")))
