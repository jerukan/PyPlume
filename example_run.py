from parcels import AdvectionRK4

from pyplume.dataloaders import DataLoader, SurfaceGrid
from pyplume.gapfilling import Gapfiller, InterpolationStep, SmoothnStep
from pyplume.kernels import AgeParticle, RandomWalk, ThreddsParticle
from pyplume.simulation import ParcelsSimulation
from pyplume.thredds_data import SRC_THREDDS_HFRNET_UCSD


ocean_data_source = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

loader = DataLoader(
    ocean_data_source,
    datasource=SRC_THREDDS_HFRNET_UCSD,
    time_range=["2020-02-09T01:00", "2020-02-14T01:00"],
    lat_range=[32.525, 32.7],
    lon_range=[-117.27, -117.09]
)

ocean_data_source_2km = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd"
ocean_data_source_6km = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

loader_2km = DataLoader(ocean_data_source_2km, datasource=SRC_THREDDS_HFRNET_UCSD)
loader_6km = DataLoader(ocean_data_source_6km, datasource=SRC_THREDDS_HFRNET_UCSD)

gapfiller = Gapfiller(
    InterpolationStep(loader_2km.data, loader_6km.data),
    SmoothnStep(mask=loader.get_mask(num_samples=50))
)

filled_ds = gapfiller.execute(target=loader.data)

ocean_grid = SurfaceGrid(filled_ds)

sim = ParcelsSimulation(
    "basic_example",
    ocean_grid,
    spawn_points=[
      [32.551707, -117.138],
      [32.557, -117.138]
    ],
    save_dir="results",
    particle_type=ThreddsParticle,
    snapshot_interval=3600,
    kernels=[AdvectionRK4, AgeParticle, RandomWalk],
    time_range=["2020-02-09T01:00", "2020-02-14T01:00"],
    repetitions=-1,
    repeat_dt=3600,
    instances_per_spawn=1,
    simulation_dt=300
)

sim.execute()
