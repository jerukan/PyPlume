from parcels import AdvectionRK4

from pyplume.dataloaders import DataLoader, SurfaceGrid
from pyplume.gapfilling import Gapfiller, LowResOversample, DCTPLS
from pyplume.kernels import AgeParticle, RandomWalk5cm, ThreddsParticle
from pyplume.resultplots import (
    ParticleWithTrackedPointsPlot,
    NearcoastDensityHistogram,
    StationTable,
)
from pyplume.simulation import ParcelsSimulation


ocean_data_source = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

loader = DataLoader(
    ocean_data_source,
    time_range=["2020-02-09T01:00", "2020-02-14T01:00"],
    lat_range=[32.525, 32.7],
    lon_range=[-117.27, -117.09],
)

ocean_data_source_2km = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd"
ocean_data_source_6km = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

loader_2km = DataLoader(ocean_data_source_2km)
loader_6km = DataLoader(ocean_data_source_6km)

gapfiller = Gapfiller(
    LowResOversample([loader_2km.dataset, loader_6km.dataset]),
    DCTPLS(mask=loader.get_mask(num_samples=50)),
)

filled_ds = gapfiller.execute(target=loader.dataset)

ocean_grid = SurfaceGrid(filled_ds)

sim = ParcelsSimulation(
    "basic_example",
    ocean_grid,
    spawn_points=[[32.551707, -117.138], [32.557, -117.138]],
    save_dir="results",
    particle_type=ThreddsParticle,
    snapshot_interval=3600,
    kernels=[AdvectionRK4, AgeParticle, RandomWalk5cm],
    time_range=["2020-02-09T01:00", "2020-02-14T01:00"],
    repetitions=-1,
    repeat_dt=3600,
    instances_per_spawn=1,
    simulation_dt=300,
)

sim.execute()

sim.parcels_result.write_data(override=True)

sim.parcels_result.add_plot(
    ParticleWithTrackedPointsPlot(
        coastline="data/coastOR2Mex_tijuana.mat", draw_currents=True, figsize=(13, 8)
    ),
    label="particleplot",
)
sim.parcels_result.add_plot(
    StationTable(
        station_points="data/wq_stposition.mat",
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
    NearcoastDensityHistogram(
        origin=[32.5567724355310, -117.130164948310],
        tracked_points="data/wq_stposition.mat",
        coastline="data/coastline.mat",
        xlim=[-16, 4],
        ymax=1,
        track_dist=900,
    ),
    label="nearcoast_density",
)

PLOT_DOMAIN = {"S": 32.525, "N": 32.7, "W": -117.27, "E": -117.09}

sim.parcels_result.generate_plots()

sim.parcels_result.generate_gifs()
print("Finished simulation")
