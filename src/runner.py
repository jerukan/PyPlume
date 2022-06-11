"""
Methods in here related to preparing, running, and processing simulations.
"""
import numpy as np
import os

from parcels import Field, VectorField
from parcels.tools.converters import GeographicPolar, Geographic

import src.utils as utils
from src.thredds_utils import rename_dataset_vars
from src.parcels_utils import HFRGrid, read_netcdf_info
from src.parcels_sim import ParcelsSimulation
from src.parcels_analysis import add_feature_set_to_result
from src.plot_features import BuoyPathFeature
from src.gapfilling import Gapfiller


empty = (None, {}, "", [])


def prep_sim_from_cfg(cfg) -> ParcelsSimulation:
    # ocean is required, no check
    ocean_cfg = utils.get_path_cfg(cfg["netcdf_data"]["ocean"])
    ds = read_netcdf_info(ocean_cfg)
    gapfiller = Gapfiller.load_from_config(*ocean_cfg.get("gapfill_steps", []))
    ds = gapfiller.execute(HFRGrid(ds))
    bound_cond = cfg["parcels_config"].get("boundary", None)
    fields = []
    # load alongshore current data if it exists
    if cfg["netcdf_data"].get("alongshore", None) not in empty:
        coast_ds = read_netcdf_info(cfg["netcdf_data"]["alongshore"])
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
    if bound_cond is None:
        grid = HFRGrid(ds, fields=fields)
    elif bound_cond.lower() in ("free", "freeslip"):
        grid = HFRGrid(
            ds, fields=fields, fs_kwargs={"interp_method": {"U": "freeslip", "V": "freeslip"}}
        )
    elif bound_cond.lower() in ("partial", "partialslip"):
        grid = HFRGrid(
            ds, fields=fields, fs_kwargs={"interp_method": {"U": "partialslip", "V": "partialslip"}}
        )
    else:
        raise ValueError(f"Invalid boundary condition {bound_cond}")
    # load wind data if it exists
    wind_data = cfg["netcdf_data"].get("wind", None)
    if wind_data not in empty:
        wind_ds = rename_dataset_vars(wind_data["path"])
        if wind_data["add_to_field_directly"]:
            grid.modify_with_wind(wind_ds, ratio=wind_data["ratio"])
        else:
            raise NotImplementedError("Wind kernel not implemented. Set add_to_field_directly to true")
    name = cfg["name"]
    print(f"Preparing simulation {name}")
    sim = ParcelsSimulation(name, grid, cfg["parcels_config"])
    return sim


def handle_postprocessing(result, postprocess_cfg):
    if postprocess_cfg.get("coastline", None) not in empty:
        lats, lons = utils.load_geo_points(**utils.get_path_cfg(postprocess_cfg["coastline"]))
        result.add_coastline(lats, lons)
        result.process_coastline_collisions()
        print("processed collisions")
    if postprocess_cfg.get("buoy", None) not in empty:
        result.add_plot_feature(
            BuoyPathFeature.from_csv(
                postprocess_cfg["buoy"],
                backstep_delta=np.timedelta64(1, "h"),
                backstep_count=12
            ), "buoy"
        )
        result.write_feature_dists(["buoy"])
        print("processed buoy distances")
    result.write_data(override=True)


def process_results(sim, cfg):
    if not sim.completed:
        raise RuntimeError("Simulation has not been completed yet.")
    postprocess_cfg = cfg.get("postprocess_config", None)
    if postprocess_cfg not in empty:
        handle_postprocessing(sim.parcels_result, postprocess_cfg)
    sim.parcels_result.write_data(override=True)
    if cfg["save_snapshots"]:
        plotting_cfg = cfg["plotting_config"]
        if plotting_cfg.get("plot_feature_set", None) not in empty:
            add_feature_set_to_result(sim.parcels_result, plotting_cfg["plot_feature_set"])
        sim.parcels_result.generate_all_plots(
            os.path.join(plotting_cfg["save_dir_snapshots"], sim.name),
            domain=plotting_cfg.get("shown_domain", None),
            land=plotting_cfg.get("draw_coasts", False),
            figsize=(13, 8)
        )
        try:
            print(sim.parcels_result.generate_gif(os.path.join(plotting_cfg["save_dir_snapshots"], f"{sim.name}.gif")))
        except FileNotFoundError:
            print("magick is not installed, gif will not be generated")
