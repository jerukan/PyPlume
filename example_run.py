from parcels import AdvectionRK4

from pyplume.dataloaders import DataLoader, SurfaceGrid
from pyplume.gapfilling import Gapfiller, LowResOversample, DCTPLS
from pyplume.kernels import AgeParticle, RandomWalk5cm, ThreddsParticle, DeleteStatusOutOfBounds
from pyplume.resultplots import (
    ParticlePlot,
    ParticleWithTrackedPointsPlot,
    NearcoastDensityHistogram,
    StationTable,
    CumulativeParticleDensityPlot
)
from pyplume.simulation import ParcelsSimulation


### Data loading ###

ocean_data_source = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

loader = DataLoader(
    ocean_data_source,
    time_range=["2020-02-09T01:00", "2020-02-14T01:00"],
    lat_range=[32.525, 32.7],
    lon_range=[-117.27, -117.09],
)

# gapfilling patchy data
ocean_data_source_2km = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd"
ocean_data_source_6km = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

gapfiller = Gapfiller(
    LowResOversample([ocean_data_source_2km, ocean_data_source_6km]),
    DCTPLS(exclude_oob=True),
)

filled_ds = gapfiller.execute(target=loader.dataset)

ocean_grid = SurfaceGrid(filled_ds)

### Simulation loading and execution ###

sim = ParcelsSimulation(
    "basic_example",
    ocean_grid,
    spawn_points=[[32.551707, -117.138], [32.557, -117.138]],
    save_dir="results",
    particle_type=ThreddsParticle,
    snapshot_interval=3600,
    kernels=[AdvectionRK4, AgeParticle, RandomWalk5cm, DeleteStatusOutOfBounds],
    time_range=["2020-02-09T01:00", "2020-02-14T01:00"],
    repetitions=-1,
    repeat_dt=3600,
    instances_per_spawn=1,
    simulation_dt=300,
    add_dir_timestamp=True,
)

sim.execute()

sim.parcels_result.to_netcdf()

### PLOTTING ###

PLOT_DOMAIN = {"S": 32.525, "N": 32.7, "W": -117.27, "E": -117.09}

sim.parcels_result.add_plot(
    ParticlePlot(
        draw_currents=True,
        coastline=True,
        particle_color="lifetime",
        domain=PLOT_DOMAIN,
        plot_size=(13, 8)
    ),
    label="particleplot",
)

station_points = [
    [  32.67780998, -117.17731484],
    [  32.63663446, -117.14441879],
    [  32.62609752, -117.1395647 ],
    [  32.58574746, -117.13296898],
    [  32.57901778, -117.13296898],
    [  32.57247503, -117.13259511],
    [  32.56555841, -117.13334285],
    [  32.56125889, -117.13166043],
    [  32.55677244, -117.13016495],
    [  32.54312613, -117.12493075],
    [  32.53527484, -117.12418301],
    [  32.50186009, -117.12441668]
]

sim.parcels_result.add_plot(
    StationTable(
        station_points=station_points,
        station_labels=[
            "Coronado (North Island)",
            "Silver Strand",
            "Silver Strand Beach",
            "Carnation Ave.",
            "Imperial Beach Pier",
            "Cortez Ave.",
            "End of Seacoast Dr.",
            "3/4 mi. N. of TJ River Mouth",
            "Tijuana River Mouth",
            "Monument Rd.",
            "Board Fence",
            "Mexico",
        ],
        track_dist=1000,
    ),
    label="station",
)
sim.parcels_result.add_plot(
    CumulativeParticleDensityPlot(
        domain=PLOT_DOMAIN,
        coastline=True
    ),
    label="cumulative_density",
)

sim.parcels_result.generate_plots()

sim.parcels_result.generate_gifs()
print("Finished simulation")
