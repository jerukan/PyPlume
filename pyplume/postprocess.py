"""
Everything in this file is related to processing the NetCDF file created by a Parcels particle
file.
"""
from pathlib import Path
from typing import Generator

import numpy as np
from shapely.geometry import LineString
from tqdm import tqdm
import xarray as xr

from pyplume import get_logger, plotting, utils
from pyplume.dataloaders import SurfaceGrid


logger = get_logger(__name__)


class ParticleResult:
    """
    Wraps the output of a particle file to make visualizing and analyzing the results easier.
    Can also use an SurfaceGrid if the ocean currents are also wanted in the plot.

    NOTE this currently only works with simulations with ThreddsParticle particle classes.
    """

    # for the main plot that draws the particles per frame
    MAIN_PARTICLE_PLOT_NAME = "particles"

    def __init__(self, src, sim_result_dir=None, snapshot_interval=None):
        """
        Args:
            src: path to ParticleFile or just the dataset itself
        """
        self.sim_result_dir = (
            Path(sim_result_dir) if sim_result_dir is not None else None
        )
        self.snapshot_interval = snapshot_interval
        if isinstance(src, (Path, str)):
            self.path = Path(src)
            self.ds = xr.open_dataset(src)
            self.ds.close()
        elif isinstance(src, xr.Dataset):
            self.path = None
            self.ds = src
        else:
            raise TypeError(f"{src} is not a path or xarray dataset")
        if self.sim_result_dir is None and self.path is not None:
            # assume particle file is already in the results directory
            self.sim_result_dir = self.path.parent
        # assumed to be in data_vars: trajectory, time, lat, lon, z
        # these variables are generated from default particles in Parcels
        self.data_vars = {}
        # data variables with different dimensions than the dataset's
        self.non_vars = {}
        self.shape = (self.ds.dims["trajectory"], self.ds.dims["obs"])
        for var, arr in self.ds.variables.items():
            arr = arr.values
            if self.shape == arr.shape:
                self.data_vars[var] = arr
            else:
                self.non_vars[var] = arr
        # some particles die in a timestamp between the time intervals, and leave a specific
        # time that probably isn't included in most of the other particles
        # we take the unique timestamps across all particles
        times_unique = np.unique(self.data_vars["time"])
        self.times = np.sort(times_unique[~np.isnan(times_unique)])

        self.grid = None
        self.frames = None
        self.plots = {}
        self.plot_paths = {}
        self.coastline = None
        # to avoid plot label conflicts
        self.counter = 0

    def add_coastline(self, lats, lons):
        """Adds a single coastline for processing collisions"""
        self.coastline = LineString(np.array([lons, lats]).T)

    def process_coastline_collisions(self):
        """
        Checks when each particle has collided with a coastline and removes all instances of the
        particle after the time of collision.
        Does not modify the original file (it shouldn't at least).
        """
        if self.coastline is None:
            raise AttributeError("Coastline is not defined yet")
        for i in range(self.data_vars["lat"].shape[0]):
            nan_where = np.where(np.isnan(self.data_vars["lat"][i]))[0]
            # LineString can't handle nan values, filter them out
            if len(nan_where) > 0 and nan_where[0] > 1:
                trajectory = LineString(
                    np.array(
                        [
                            self.data_vars["lon"][i][: nan_where[0]],
                            self.data_vars["lat"][i][: nan_where[0]],
                        ]
                    ).T
                )
            elif len(nan_where) == 0:
                trajectory = LineString(
                    np.array([self.data_vars["lon"][i], self.data_vars["lat"][i]]).T
                )
            else:
                # found an all nan particle (somehow)
                continue
            if trajectory.intersects(self.coastline):
                # the entire trajectory intersects with the coastline, find which timestamp it
                # crosses and delete all data after that
                for j in range(1, self.data_vars["lat"].shape[1]):
                    if np.isnan(self.data_vars["lon"][i, j]):
                        break
                    part_seg = LineString(
                        [
                            (
                                self.data_vars["lon"][i, j - 1],
                                self.data_vars["lat"][i, j - 1],
                            ),
                            (self.data_vars["lon"][i, j], self.data_vars["lat"][i, j]),
                        ]
                    )
                    if self.coastline.intersects(part_seg):
                        for var in self.data_vars.keys():
                            if np.issubdtype(self.data_vars[var].dtype, np.datetime64):
                                self.data_vars[var][i, j:] = np.datetime64("NaT")
                            else:
                                self.data_vars[var][i, j:] = np.nan
                        break

    def add_grid(self, grid: SurfaceGrid):
        """Adds a SurfaceGrid to draw the currents on the plots."""
        self.grid = grid
        gtimes, _, _ = grid.get_coords()
        # check if particle set is in-bounds of the given grid
        if (
            np.nanmin(self.data_vars["time"]) < gtimes.min()
            or np.nanmax(self.data_vars["time"]) > gtimes.max()
        ):
            # TODO time extrapolation
            # raise ValueError("Time out of bounds")
            pass

    # TODO finish
    def plot_trajectory(self, idxs, domain=None, land=True):
        if not isinstance(idxs, list):
            idxs = [idxs]
        plotting.draw_trajectories

    def add_plot(self, resultplot, label=None):
        if label is None:
            label = f"{resultplot.__class__.__name__}_{self.counter}"
            self.counter += 1
        self.plots[label] = resultplot
        self.plot_paths[label] = []

    def generate_plots(self, clear_folder=False):
        if self.sim_result_dir is None:
            raise ValueError(
                "Please specify a path for sim_result_dir to save the plots"
            )
        for label, resultplot in self.plots.items():
            save_dir = self.sim_result_dir / "plots"
            utils.get_dir(save_dir)
            if clear_folder:
                utils.delete_all_pngs(save_dir)
            allplots = resultplot(self)
            if isinstance(allplots, Generator):
                for i, (fig, ax) in tqdm(
                    enumerate(allplots), desc=f"Generating plots for {label}"
                ):
                    savefile = utils.get_dir(save_dir / label) / f"simframe_{i}.png"
                    self.plot_paths[label].append(savefile)
                    plotting.draw_plt(savefile=savefile, fig=fig)
            else:
                figs, axs = allplots
                for i, fig in tqdm(
                    enumerate(figs), desc=f"Generating plots for {label}"
                ):
                    savefile = utils.get_dir(save_dir / label) / f"simframe_{i}.png"
                    self.plot_paths[label].append(savefile)
                    plotting.draw_plt(savefile=savefile, fig=fig)
            print(f"Completed generating plots for {label} inside {save_dir}")
            logger.info(f"Completed generating plots for {label} inside {save_dir}")

    def get_plot_timestamps(self):
        plot_times = []
        if self.snapshot_interval is not None:
            # The delta time between each snapshot is defined in the parcels config.
            # This lets us avoid the in-between timestamps where a single particle
            # gets deleted.
            guessed_interval = self.snapshot_interval
        else:
            # Guess the time interval if it's not given.
            diffs = np.diff(self.times)
            diff_vals, diff_counts = np.unique(diffs, return_counts=True)
            idx_max = np.argmax(diff_counts)
            guessed_interval = diff_vals[idx_max]
        t = self.times[0]
        while t <= self.times[-1]:
            plot_times.append(t)
            t += np.timedelta64(guessed_interval, "s")
        # sometimes the final timestamp is an incomplete interval
        # add it directly if it happens
        if self.times[-1] != plot_times[-1]:
            plot_times.append(self.times[-1])
        return np.array(plot_times)

    def generate_gifs(self, frame_duration=500):
        """Uses imagemagick to generate a gif of the main simulation plot."""
        for label, resultplot_paths in self.plot_paths.items():
            gif_path = self.sim_result_dir / f"{label}.gif"
            input_paths = [str(path) for path in resultplot_paths]
            utils.generate_gif(input_paths, gif_path, frame_duration=frame_duration)

    def to_netcdf(self, path=None):
        """
        Write postprocessed data to a new netcdf file.
        If path is unspecified, it will override the original path the ParticleFile was
        saved to (if a path was passed in).
        """
        if path is None:
            path = Path(self.path)
            path = path.parent / f"{path.stem}.nc"
        if self.path is None and path is None:
            raise ValueError("No path specified and no previous path was passed in.")
        self.ds.to_netcdf(path=path)

    def get_filtered_data_time(self, t: np.datetime64, query="at"):
        if isinstance(t, int):
            t = self.times[t]
        valid_queries = ("at", "before", "after")
        if query not in valid_queries:
            raise ValueError(
                f"Invalid query '{query}'. Valid queries are {valid_queries}"
            )
        if query == "at":
            mask = self.data_vars["time"] == t
        elif query == "before":
            mask = self.data_vars["time"] <= t
        elif query == "after":
            mask = self.data_vars["time"] >= t
        filtered = {}
        for dvar, dval in self.data_vars.items():
            filtered[dvar] = dval[mask]
        return filtered

    def get_positions_time(self, t: np.datetime64, query="at"):
        filtered = self.get_filtered_data_time(t, query=query)
        return filtered["lat"], filtered["lon"]


class ParticleResultComparer:
    def __init__(self, *particleresults):
        self.particleresults = particleresults
