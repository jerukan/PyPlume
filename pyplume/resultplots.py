from typing import Generator

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from pyplume import plotting, dataloaders, utils
from pyplume.postprocess import ParticleResult


class StaticAddOn:
    def add_to_plot(self, fig, ax):
        raise NotImplementedError()


class AddScatter(StaticAddOn):
    def __init__(self, data, color=None, size=None, edgecolor=None):
        self.lats, self.lons = dataloaders.load_geo_points(data)
        self.color = color
        self.size = size
        self.edgecolor = edgecolor

    def add_to_plot(self, fig, ax):
        ax.scatter(
            self.lons, self.lats, c=self.color, s=self.size, edgecolor=self.edgecolor
        )
        return fig, ax


class ResultPlot:
    def __init__(self, plot_size=None):
        self.addons = []
        self.plot_size = plot_size

    def add_addon(self, addon: StaticAddOn):
        self.addons.append(addon)

    def generate_plots(self, result: ParticleResult):
        raise NotImplementedError()

    def _generate_plots(self, result: ParticleResult):
        """Generates all plots with addons."""
        allplots = self.generate_plots(result)
        if isinstance(allplots, Generator):
            for fig, ax in allplots:
                for addon in self.addons:
                    addon.add_to_plot(fig, ax)
                if self.plot_size is not None:
                    fig.set_size_inches(self.plot_size)
                yield fig, ax
        else:
            figs, axs = allplots
            for fig, ax in zip(figs, axs):
                for addon in self.addons:
                    addon.add_to_plot(fig, ax)
                if self.plot_size is not None:
                    fig.set_size_inches(self.plot_size)
            return figs, axs

    def __call__(self, result: ParticleResult):
        return self._generate_plots(result)


class ParticlePlot(ResultPlot):
    def __init__(
        self,
        particle_size=None,
        domain=None,
        coastline=None,
        draw_currents=False,
        color_currents=False,
        particle_color=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.particle_size = particle_size
        self.domain = domain
        # only if coastline is the boolean true, otherwise we try to parse it
        self.coastline = None
        if coastline is True:
            self.carree_land = True
        elif coastline is False or coastline is None:
            self.carree_land = False
        else:
            self.coastline = dataloaders.load_geo_points(coastline)
            self.carree_land = False
        self.draw_currents = draw_currents
        self.color_currents = color_currents
        self.particle_color = particle_color

    def plot_at_time(self, time, result: ParticleResult):
        if self.draw_currents:
            if result.grid is None:
                raise ValueError(
                    "ParticleResult needs a loaded grid to plot the vector field"
                )
            fig, ax = plotting.plot_vectorfield(
                result.grid.dataset,
                show_time=time,
                domain=self.domain,
                land=self.carree_land,
                allow_time_extrapolation=True,
                color_speed=self.color_currents,
                cbar=True,
            )
        else:
            fig, ax = plotting.get_carree_axis(
                domain=self.domain, land=self.carree_land
            )
        data_t = result.get_filtered_data_time(time)
        lats = data_t["lat"]
        lons = data_t["lon"]
        color_data = None
        color_max = None
        color_min = None
        cbar_label = None
        if self.particle_color is not None:
            color_data = data_t[self.particle_color]
            color_min = np.nanmin(result.data_vars[self.particle_color])
            color_max = np.nanmax(result.data_vars[self.particle_color])
            if self.particle_color == "lifetime":
                color_data /= 86400
                color_min /= 86400
                color_max /= 86400
                cbar_label = "Age (days)"
            else:
                cbar_label = self.particle_color
        fig, ax = plotting.plot_particles(
            lats,
            lons,
            color=color_data,
            edgecolor="k",
            vmin=color_min,
            vmax=color_max,
            size=self.particle_size,
            cbar=color_data is not None,
            cbar_label=cbar_label,
            ax=ax,
        )
        if self.coastline is not None:
            plotting.plot_coastline(self.coastline[0], self.coastline[1], ax=ax)
        ax.set_title(f"Particles at {time}")
        return (fig, ax), data_t

    def generate_plots(self, result: ParticleResult):
        times = result.get_plot_timestamps()
        for t in times:
            (fig, ax), _ = self.plot_at_time(t, result)
            yield fig, ax


class ParticleWithTrackedPointsPlot(ParticlePlot):
    def __init__(self, tracked_points, track_dist, **kwargs):
        super().__init__(**kwargs)
        self.lats, self.lons = dataloaders.load_geo_points(tracked_points)
        self.ptcollection = utils.GeoPointCollection(self.lats, self.lons)
        self.track_dist = track_dist

    def generate_plots(self, result: ParticleResult):
        times = result.get_plot_timestamps()
        for t in times:
            (fig, ax), data = self.plot_at_time(t, result)
            # plot tracked points
            lats = data["lat"]
            lons = data["lon"]
            counts = self.ptcollection.count_near(lats, lons, self.track_dist)
            ax.scatter(
                self.lons[counts == 0],
                self.lats[counts == 0],
                marker="^",
                c="b",
                s=60,
                edgecolor="k",
                zorder=3,
            )
            ax.scatter(
                self.lons[counts > 0],
                self.lats[counts > 0],
                marker="^",
                c="r",
                s=60,
                edgecolor="k",
                zorder=3,
            )
            ax.set_title(f"Particles at {t}")
            yield fig, ax


class NearcoastDensityHistogram(ResultPlot):
    def __init__(
        self,
        origin=None,
        tracked_points=None,
        track_dist=None,
        coastline=None,
        xlim=None,
        ymax=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.origin_lat, self.origin_lon = dataloaders.load_geo_points(origin)
        self.origin_collection = utils.GeoPointCollection(
            self.origin_lat, self.origin_lon
        )
        self.tracked_lats, self.tracked_lons = dataloaders.load_geo_points(
            tracked_points
        )
        self.track_dist = track_dist
        self.coast_lats, self.coast_lons = dataloaders.load_geo_points(coastline)
        self.coastline = LineString(np.array([self.coast_lons, self.coast_lats]).T)
        self.xlim = xlim
        self.ymax = ymax

    def generate_plots(self, result: ParticleResult):
        times = result.get_plot_timestamps()
        for t in times:
            data_t = result.get_filtered_data_time(t)
            lats = data_t["lat"]
            lons = data_t["lon"]
            coast_dists = np.empty(len(lats))
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                _, coast_nearest = nearest_points(Point(lon, lat), self.coastline)
                coast_dists[i] = utils.haversine(
                    coast_nearest.y, lat, coast_nearest.x, lon
                )
            dists = self.origin_collection.get_all_dists(lats, lons)[0]
            station_dists = self.origin_collection.get_all_dists(
                self.tracked_lats, self.tracked_lons
            )[0]
            # things north of the origin will appear on the left
            # calculate station distances
            stations_north = self.tracked_lats > self.origin_lat[0]
            station_dists[stations_north] = -station_dists[stations_north]
            station_dists /= 1000
            # find which particles are north relative to origin and set them negative
            north = lats > self.origin_lat[0]
            dists[north] = -dists[north]
            dists /= 1000
            nearcoast = coast_dists <= self.track_dist
            if self.xlim is None:
                xlim = [dists[nearcoast].min(), dists[nearcoast].max()]
            else:
                xlim = self.xlim
            # hack to prevent non-nearcoast particles from showing
            dists[~nearcoast] = xlim[1] + 1
            fig = plt.figure()
            ax = fig.add_subplot()
            bins = np.linspace(xlim[0], xlim[1], 30)
            bins = np.append(bins, self.xlim[1] + 1)
            ax.hist(dists, bins=bins, density=True)
            ax.scatter(
                x=station_dists,
                y=np.full(station_dists.shape, 0.01),
                c="k",
                edgecolor="y",
                zorder=1000,
            )
            ax.set_xlim(xlim)
            if self.ymax is not None:
                ax.set_ylim([0, self.ymax])
            # generate tick labels
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticks())
            # matplotlib uses a funny hyphen that doesn't work
            labels = list(map(plotting.abs_label_map, ax.get_xticklabels()))
            ax.set_xticklabels(labels)
            plt.figtext(
                0.5,
                -0.01,
                "(North) ------ Distance from point (km) ------ (South)",
                horizontalalignment="center",
            )
            fig.set_size_inches(6.1, 2.5)
            yield fig, ax


class StationTable(ResultPlot):
    def __init__(
        self, station_points=None, station_labels=None, track_dist=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.lats, self.lons = dataloaders.load_geo_points(station_points)
        self.ptcollection = utils.GeoPointCollection(self.lats, self.lons)
        self.station_labels = station_labels
        self.track_dist = track_dist

    def generate_plots(self, result: ParticleResult):
        times = result.get_plot_timestamps()
        for t in times:
            lats, lons = result.get_positions_time(t)
            colors = np.full((len(self.lats), 4), "white", dtype=object)
            counts = self.ptcollection.count_near(lats, lons, self.track_dist).astype(
                np.uint32
            )
            for i in range(len(self.lats)):
                if counts[i] > 0:
                    colors[i, :] = "lightcoral"
            plume_pot = np.where(counts > 0, "YES", "NO")
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_axis_off()
            ax.table(
                cellText=np.array(
                    [np.arange(len(counts)) + 1, self.station_labels, counts, plume_pot]
                ).T,
                cellColours=colors,
                colLabels=[
                    "Station ID",
                    "Station Name",
                    "Particle Count",
                    "Plume Potential",
                ],
                loc="center",
            ).auto_set_column_width(col=[0, 1, 2, 3, 4])
            ax.axis("tight")
            # fig.set_size_inches(7.17, 4)
            yield fig, ax


class CumulativeParticleDensityPlot(ResultPlot):
    def __init__(self, domain=None, coastline=None, bins=None, pmax=None, **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.cartopy_coastline = False
        if isinstance(coastline, bool):
            self.cartopy_coastline = coastline
        else:
            self.coast_lats, self.coast_lons = dataloaders.load_geo_points(coastline)
        self.pmax = pmax
        self.bins = bins

    def generate_plots(self, result: ParticleResult):
        times = result.get_plot_timestamps()
        for t in times:
            lats, lons = result.get_positions_time(t, query="before")
            fig, ax = plotting.plot_particle_density(
                lats,
                lons,
                domain=self.domain,
                bins=self.bins,
                pmax=self.pmax,
                land=self.cartopy_coastline,
                title=f"Cumilative particle density at {t}",
            )
            if not self.cartopy_coastline:
                plotting.plot_coastline(self.coast_lats, self.coast_lons, ax=ax)
            yield fig, ax
