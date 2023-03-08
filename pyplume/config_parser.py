"""
Methods in here related to preparing, running, and processing simulations.
"""
import copy
from datetime import datetime
import json
import logging
import os
from pathlib import Path

import numpy as np
from parcels import Field, VectorField
from parcels.tools.converters import GeographicPolar, Geographic
import xarray as xr
import yaml

from pyplume import get_logger
from pyplume.constants import EMPTY
from pyplume.dataloaders import DataSource, DataLoader, DefaultLoad, SurfaceGrid, load_wind_dataset, load_geo_points
import pyplume.utils as utils
from pyplume.simulation import ParcelsSimulation
from pyplume.plot_features import BuoyPathFeature, construct_features_from_configs
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
    dsource = cfg.pop("datasource", None)
    if dsource is None:
        u_key = cfg.pop("u_key", None)
        v_key = cfg.pop("v_key", None)
        if u_key is None and v_key is None:
            uv_map = None
        elif u_key is not None and v_key is not None:
            uv_map = {"U": u_key, "V": v_key}
        else:
            raise ValueError("You cannot only specify either a U or V key, you must define both.")
        time_key = cfg.pop("time_key", None)
        lat_key = cfg.pop("lat_key", None)
        lon_key = cfg.pop("lon_key", None)
        if time_key is None and lat_key is None and lon_key is None:
            coord_map = None
        elif time_key is not None and lat_key is not None and lon_key is not None:
            coord_map = {"time": time_key, "lat": lat_key, "lon": lon_key}
        else:
            raise ValueError("You cannot only specify either a time, lat, or lon key, you must define them all.")
        drop_vars = cfg.pop("drop_vars", None)
        dsource = DataSource(id="idk", name="idk", load_method=DefaultLoad(uv_map=uv_map, coord_map=coord_map, drop_vars=drop_vars))
    ds = DataLoader(ds_path, datasource=dsource, **cfg).dataset
    gapfiller = Gapfiller.load_from_config(*cfg.get("gapfill_steps", []))
    ds = gapfiller.execute(ds)
    fields = []
    # load alongshore current data if it exists
    if cfg.get("alongshore", None) not in EMPTY:
        alongshore_cfg = utils.get_path_cfg(cfg["alongshore"])
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
    # set boundary condition of fields
    allow_time_extrapolation = cfg.get("allow_time_extrapolation", False)
    fs_kwargs = {"allow_time_extrapolation": allow_time_extrapolation}
    grid = SurfaceGrid(ds, other_fields=fields, boundary_condition=boundary_condition, **fs_kwargs)
    # load wind data if it exists
    wind_cfg = cfg.get("wind", None)
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
    parcels_cfg["save_dir"] = Path(parcels_cfg["save_dir"]) / f"{simset_name}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    sims = []
    # ocean is required, no check
    ocean_cfgs = cfg["ocean_data"]
    if isinstance(ocean_cfgs, dict):
        ocean_cfgs = [ocean_cfgs]
    elif isinstance(ocean_cfgs, str):
        ocean_cfgs = [utils.wrap_in_kwarg(ocean_cfgs, key="data")]
    for ocean_cfg in ocean_cfgs:
        ocean_cfg = utils.wrap_in_kwarg(ocean_cfg, key="data")
        name = ocean_cfg.get("name", None)
        if name is None:
            raise ValueError("ocean_data config needs a name")
        grid = load_ocean_cfg(ocean_cfg)
        sim = ParcelsSimulation(name, grid, **parcels_cfg)
        logger.info(f"Preparing simulation {name}")
        sims.append(sim)
    return sims


def handle_postprocessing(result, postprocess_cfg):
    if postprocess_cfg.get("coastline", None) not in EMPTY:
        lats, lons = load_geo_points(
            **utils.get_path_cfg(postprocess_cfg["coastline"])
        )
        result.add_coastline(lats, lons)
        result.process_coastline_collisions()
        logger.info("processed collisions")
    if postprocess_cfg.get("buoy", None) not in EMPTY:
        result.add_plot_feature(
            BuoyPathFeature.load_from_external(
                postprocess_cfg["buoy"],
                backstep_delta=np.timedelta64(1, "h"),
                backstep_count=12,
            ),
            label="buoy",
        )
        result.write_feature_dists(["buoy"])
        logger.info("processed buoy distances")


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
            figsize=(13, 8),
        )
        try:
            logger.info(sim.parcels_result.generate_gif())
        except FileNotFoundError:
            logger.info("magick is not installed, gif will not be generated")
