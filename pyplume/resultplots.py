import matplotlib.pyplot as plt
import numpy as np

from pyplume import plotting, dataloaders
from pyplume.postprocess import ParticleResult


class ResultPlotAddOn:

    def add_to_plots(self, figs, axs):
        return figs, axs


class AddScatter(ResultPlotAddOn):

    def __init__(self, data, color=None, size=None, edgecolor=None):
        self.lats, self.lons = dataloaders.load_geo_points(data)
        self.color = color
        self.size = size
        self.edgecolor = edgecolor

    def add_to_plots(self, figs, axs):
        for ax in axs:
            ax.scatter(self.lons, self.lats, c=self.color, s=self.size, edgecolor=self.edgecolor)
        return figs, axs


class ResultPlot:

    def __init__(self):
        self.addons = []

    def add_addon(self, addon: ResultPlotAddOn):
        self.addons.append(addon)

    def generate_plots(self, result: ParticleResult):
        figs = []
        axs = []
        return figs, axs

    def _generate_plots(self, result: ParticleResult):
        figs, axs = self.generate_plots(result)
        for addon in self.addons:
            figs, axs = addon.add_to_plots(figs, axs)
        return figs, axs
    

class ParticlePlot(ResultPlot):

    def __init__(self, particle_size=None, domain=None, coastline=None, draw_currents=False, show_ages=False):
        super().__init__()
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
        self.show_ages = show_ages

    def generate_plots(self, result: ParticleResult):
        figs = []
        axs = []
        times = result.get_plot_timestamps()
        for t in times:
            if self.draw_currents:
                if result.grid is None:
                    raise ValueError("ParticleResult needs a loaded grid to plot the vector field")
                fig, ax = plotting.plot_vectorfield(
                    result.grid.dataset, show_time=t, domain=self.domain, land=self.carree_land
                )
            else:
                fig, ax = plotting.get_carree_axis(domain=self.domain, land=self.carree_land)
            data_t = result.get_filtered_data_time(t)
            lats = data_t["lat"]
            lons = data_t["lon"]
            lifetimes = None
            lifetime_max = None
            lifetime_min = None
            if self.show_ages:
                lifetimes = data_t["lifetime"]
                lifetime_max = np.nanmax(result.data_vars["lifetime"]) / 86400
                lifetime_min = 0
            sc = ax.scatter(
                lons,
                lats,
                c=lifetimes,
                edgecolor="k",
                vmin=lifetime_min,
                vmax=lifetime_max,
                s=self.particle_size,
            )
            if lifetimes is not None:
                age_cbar = plt.colorbar(sc)
                age_cbar.set_label("Age (days)")
            if self.coastline is not None:
                plotting.plot_coastline(self.coastline[0], self.coastline[1], ax=ax)
            # figs.append(fig)
            # axs.append(ax)
            yield fig, ax
        # return figs, axs
