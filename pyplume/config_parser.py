"""
Methods in here related to preparing, running, and processing simulations
through the use of the YAML configs.
"""
import copy
from datetime import datetime
from pathlib import Path

from parcels import Field, VectorField
from parcels.tools.converters import GeographicPolar, Geographic
import yaml

from pyplume import get_logger
from pyplume.constants import EMPTY
from pyplume.dataloaders import (
    DataLoader,
    SurfaceGrid,
    load_wind_dataset,
    load_geo_points,
)
import pyplume.utils as utils
from pyplume.simulation import ParcelsSimulation
from pyplume.gapfilling import Gapfiller


logger = get_logger(__name__)


def load_config(path):
    """
    Returns a json file as a dict.

    Args:
        path (str)

    Returns:
        dict: data pulled from the json specified.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if config.get("name", None) is None:
        config["name"] = Path(path).stem
    # TODO do some config verification here
    return config


def load_ocean_cfg(cfg):
    ds_path = cfg["data"]
    del cfg["data"]
    boundary_condition = cfg.pop("boundary_condition", None)
    alongshore = cfg.pop("alongshore", None)
    allow_time_extrapolation = cfg.pop("allow_time_extrapolation", False)
    wind_cfg = cfg.pop("wind", None)
    gapfiller = Gapfiller.load_from_config(*cfg.pop("gapfill_steps", []))
    ds = DataLoader(ds_path, **cfg).dataset
    ds = gapfiller.execute(ds)
    fields = []
    # load alongshore current data if it exists
    if alongshore not in EMPTY:
        alongshore_cfg = utils.get_path_cfg(alongshore)
        alongshore_cfg["dataset"] = alongshore_cfg["path"]
        del alongshore_cfg["path"]
        coast_ds = DataLoader(**alongshore_cfg).dataset
        fu = Field.from_xarray(
            coast_ds["U"],
            "CU",
            dict(lat="lat", lon="lon", time="time"),
            interp_method="nearest",
        )
        fu.units = GeographicPolar()
        fv = Field.from_xarray(
            coast_ds["V"],
            "CV",
            dict(lat="lat", lon="lon", time="time"),
            interp_method="nearest",
        )
        fv.units = Geographic()
        fuv = VectorField("CUV", fu, fv)
        fields = [fuv]
    fs_kwargs = {"allow_time_extrapolation": allow_time_extrapolation}
    # set boundary condition of fields
    grid = SurfaceGrid(
        ds, other_fields=fields, boundary_condition=boundary_condition, **fs_kwargs
    )
    # load wind data if it exists
    if wind_cfg not in EMPTY:
        # TODO fix this hardcoding help oh god
        wind_path = wind_cfg["data"]
        del wind_cfg["data"]
        wind_ds = load_wind_dataset(wind_path, **wind_cfg)
        grid.modify_with_wind(wind_ds, ratio=wind_cfg["ratio"])
    return grid


def prep_sim_from_cfg(cfg):
    simset_name = cfg["name"]
    parcels_cfg = cfg["parcels_config"]
    parcels_cfg["save_dir"] = (
        Path(parcels_cfg.get("save_dir", "results"))
        / f"{simset_name}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    )
    parcels_cfg["save_dir"].mkdir(parents=True)
    sims = []
    # ocean is required, no check
    ocean_cfgs = cfg["ocean_data"]
    if isinstance(ocean_cfgs, dict):
        ocean_cfgs = [ocean_cfgs]
    elif isinstance(ocean_cfgs, str):
        ocean_cfgs = [utils.wrap_in_kwarg(ocean_cfgs, key="data")]
    for ocean_cfg in ocean_cfgs:
        ocean_cfg = utils.wrap_in_kwarg(ocean_cfg, key="data")
        name = ocean_cfg.pop("name", None)
        if name is None:
            raise ValueError("ocean_data config needs a name")
        grid = load_ocean_cfg(ocean_cfg)
        sim = ParcelsSimulation(name, grid, **parcels_cfg)
        logger.info(f"Prepared simulation {name}")
        ds_path = sim.sim_result_dir / "ocean_dataset_modified.nc"
        grid.dataset.to_netcdf(ds_path)
        print(f"Modified ocean dataset netcdf saved to {ds_path}.")
        logger.info(f"Modified ocean dataset netcdf saved to {ds_path}.")
        sims.append(sim)
    return sims


def handle_postprocessing(result, postprocess_cfg):
    if postprocess_cfg.get("coastline", None) not in EMPTY:
        lats, lons = load_geo_points(**utils.get_path_cfg(postprocess_cfg["coastline"]))
        result.add_coastline(lats, lons)
        result.process_coastline_collisions()
        logger.info("processed collisions")
    # if postprocess_cfg.get("buoy", None) not in EMPTY:
    #     result.add_plot_feature(
    #         BuoyPathFeature.load_from_external(
    #             postprocess_cfg["buoy"],
    #             backstep_delta=np.timedelta64(1, "h"),
    #             backstep_count=12,
    #         ),
    #         label="buoy",
    #     )
    #     result.write_feature_dists(["buoy"])
    #     logger.info("processed buoy distances")


def process_results(sim, cfg):
    if not sim.completed:
        raise RuntimeError("Simulation has not been completed yet.")
    postprocess_cfg = cfg.get("postprocess_config", None)
    if postprocess_cfg not in EMPTY:
        handle_postprocessing(sim.parcels_result, postprocess_cfg)
    sim.parcels_result.to_netcdf()
    if cfg.get("plotting_config", None) not in EMPTY:
        plotting_cfg = copy.deepcopy(cfg["plotting_config"])
        resultplots = plotting_cfg.get("plots", [])
        for resultplot_cfg in resultplots:
            resultplot_class = utils.import_attr(resultplot_cfg.pop("type"))
            resultplot_label = resultplot_cfg.pop("label", None)
            resultplot_addons = resultplot_cfg.pop("addons", [])
            resultplot = resultplot_class(**resultplot_cfg)
            for addon_cfg in resultplot_addons:
                addon_class = utils.import_attr(addon_cfg.pop("type"))
                addon = addon_class(**addon_cfg)
                resultplot.add_addon(addon)
            sim.parcels_result.add_plot(resultplot, label=resultplot_label)
        sim.parcels_result.generate_plots()
        logger.info(sim.parcels_result.generate_gifs())
