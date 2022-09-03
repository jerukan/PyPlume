"""
Methods in here related to preparing, running, and processing simulations.
"""
import logging
import os

import numpy as np
from parcels import Field, VectorField
from parcels.tools.converters import GeographicPolar, Geographic

from pyplume.constants import EMPTY
from pyplume.dataloaders import DataLoader, SurfaceGrid, rename_dataset_vars
import pyplume.utils as utils
from pyplume.simulation import ParcelsSimulation
from pyplume.plot_features import BuoyPathFeature, construct_features_from_configs
from pyplume.gapfilling import Gapfiller


logger = logging.getLogger(__name__)


def prep_sim_from_cfg(cfg) -> ParcelsSimulation:
    # ocean is required, no check
    ocean_cfg = utils.get_path_cfg(cfg["netcdf_data"]["ocean"])
    ocean_cfg["dataset"] = ocean_cfg["path"]
    del ocean_cfg["path"]
    boundary_condition = ocean_cfg.pop("boundary_condition", None)
    ds = DataLoader(**ocean_cfg).dataset
    gapfiller = Gapfiller.load_from_config(*ocean_cfg.get("gapfill_steps", []))
    ds = gapfiller.execute(ds)
    fields = []
    # load alongshore current data if it exists
    if cfg["netcdf_data"].get("alongshore", None) not in EMPTY:
        alongshore_cfg = utils.get_path_cfg(cfg["netcdf_data"]["alongshore"])
        alongshore_cfg["dataset"] = alongshore_cfg["path"]
        del alongshore_cfg["path"]
        coast_ds = DataLoader(**alongshore_cfg).dataset
        fu = Field.from_xarray(
            coast_ds["U"], "CU", dict(lat="lat", lon="lon", time="time"), interp_method="nearest"
        )
        fu.units = GeographicPolar()
        fv = Field.from_xarray(
            coast_ds["V"], "CV", dict(lat="lat", lon="lon", time="time"), interp_method="nearest"
        )
        fv.units = Geographic()
        fuv = VectorField("CUV", fu, fv)
        fields = [fuv]
    # set boundary condition of fields
    grid = SurfaceGrid(ds, other_fields=fields, boundary_condition=boundary_condition)
    # load wind data if it exists
    wind_data = cfg["netcdf_data"].get("wind", None)
    if wind_data not in EMPTY:
        wind_ds = rename_dataset_vars(wind_data["path"])
        if wind_data["add_to_field_directly"]:
            grid.modify_with_wind(wind_ds, ratio=wind_data["ratio"])
        else:
            raise NotImplementedError("Wind kernel not implemented. Set add_to_field_directly to true")
    name = cfg["name"]
    logger.info(f"Preparing simulation {name}")
    sim = ParcelsSimulation(name, grid, **cfg["parcels_config"])
    return sim


def handle_postprocessing(result, postprocess_cfg):
    if postprocess_cfg.get("coastline", None) not in EMPTY:
        lats, lons = utils.load_geo_points(**utils.get_path_cfg(postprocess_cfg["coastline"]))
        result.add_coastline(lats, lons)
        result.process_coastline_collisions()
        logger.info("processed collisions")
    if postprocess_cfg.get("buoy", None) not in EMPTY:
        result.add_plot_feature(
            BuoyPathFeature.load_from_external(
                postprocess_cfg["buoy"],
                backstep_delta=np.timedelta64(1, "h"),
                backstep_count=12
            ), label="buoy"
        )
        result.write_feature_dists(["buoy"])
        logger.info("processed buoy distances")
    result.write_data(override=True)


def process_results(sim, cfg):
    if not sim.completed:
        raise RuntimeError("Simulation has not been completed yet.")
    postprocess_cfg = cfg.get("postprocess_config", None)
    if postprocess_cfg not in EMPTY:
        handle_postprocessing(sim.parcels_result, postprocess_cfg)
    sim.parcels_result.write_data(override=True)
    if cfg["save_snapshots"]:
        plotting_cfg = cfg["plotting_config"]
        feature_cfgs = plotting_cfg.get("plot_features", None)
        if feature_cfgs not in EMPTY:
            features, labels = construct_features_from_configs(*feature_cfgs)
            for feature, label in zip(features, labels):
                sim.parcels_result.add_plot_feature(feature, label=label)
        sim.parcels_result.generate_all_plots(
            domain=plotting_cfg.get("shown_domain", None),
            land=plotting_cfg.get("draw_coasts", False),
            figsize=(13, 8)
        )
        try:
            logger.info(sim.parcels_result.generate_gif())
        except FileNotFoundError:
            logger.info("magick is not installed, gif will not be generated")
