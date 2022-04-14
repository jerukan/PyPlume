from abc import ABC, abstractmethod
import importlib
import os
from typing import Tuple

import matlab.engine
import numpy as np
import xarray as xr

import src.utils as utils
from src.parcels_utils import HFRGrid
import src.thredds_utils as thredds_utils


class GapfillStep(ABC):
    @abstractmethod
    def process(
        self, u: np.ndarray, v: np.ndarray, target: HFRGrid, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


def get_interped(i, target, ref, invalid_where):
    """
    Args:
        i (int): index on invalid_where
        ref (HFRGrid): reference Dataset
        invalid_where (array-like): (3, n) dimensional array representing all invalid positions
    
    Returns:
        (u, v): (nan, nan) if no data was found, interpolated values otherwise
    """
    time_diff = np.diff(ref.fieldset_flat.U.grid.time)[0]
    t = invalid_where[0][i]
    lat = target.lats[invalid_where[1][i]]
    lon = target.lons[invalid_where[2][i]]
    current_u, current_v = ref.get_fs_current(t * time_diff, lat, lon)
    current_abs = abs(current_u) + abs(current_v)
    # if both the u and v components are 0, there's probably no data there
    if np.isnan(ref.get_closest_current(t, lat, lon)[0]) or current_abs == 0:
        return np.nan, np.nan
    return current_u, current_v


class InterpolationStep(GapfillStep):
    """
    Uses linear interpolation
    """
    def __init__(self, references=None):
        self.references = references if references is not None else []

    def do_validation(self, target, loaded_references):
        targ_times, targ_lats, targ_lons = target.get_coords()
        targ_min = (targ_lats[0], targ_lons[0])
        targ_max = (targ_lats[-1], targ_lons[-1])
        # check references
        for ref in loaded_references:
            ref_times, ref_lats, ref_lons = ref.get_coords()
            lat_inbounds = (ref_lats[0] <= targ_min[0]) and (ref_lats[-1] >= targ_max[0])
            lon_inbounds = (ref_lons[0] <= targ_min[1]) and (ref_lons[-1] >= targ_max[1])
            time_inbounds = (ref_times[0] <= targ_times[0]) and (ref_times[-1] >= targ_times[-1])
            if not (lat_inbounds and lon_inbounds and time_inbounds):
                raise ValueError("Incorrect reference dimensions (reference dimension ranges \
                    should be larger than the target's)")

    def process(
        self, u: np.ndarray, v: np.ndarray, target: HFRGrid, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        loaded_references = []
        for i, ref in enumerate(self.references):
            if isinstance(ref, HFRGrid):
                loaded_references.append(ref)
            elif isinstance(ref, (str, int)):
                if os.path.isfile(ref):
                    loaded_references.append(HFRGrid(ref))
                else:
                    # assume url, attempt to open with OPENDAP
                    times, lats, lons = target.get_coords()
                    time_range = (times[0], times[-1])
                    lat_range = (lats[0], lats[-1])
                    lon_range = (lons[0], lons[-1])
                    # slice the data before loading into HFRGrid since it's huge
                    ds = thredds_utils.slice_dataset(
                        ref, time_range, lat_range, lon_range, inclusive=True
                    )
                    loaded_references.append(HFRGrid(ds))
            else:
                raise TypeError(f"Unrecognized type for {ref}")
                        
        self.do_validation(target, loaded_references)
        invalid = utils.generate_mask_invalid(u)
        num_invalid = invalid.sum()
        print(f"total invalid values on target data: {num_invalid}")

        # linear interpolation from lower resolution data
        target_interped_u = u.copy()
        target_interped_v = v.copy()
        invalid_interped = invalid.copy()
        for ref in loaded_references:
            invalid_pos_new = np.where(invalid_interped)
            num_invalid_new = int(invalid_interped.sum())
            arr_u = np.zeros(num_invalid_new)
            arr_v = np.zeros(num_invalid_new)
            print(f"Attempting to interpolate {num_invalid_new} points...")
            for i in range(num_invalid_new):
                c_u, c_v = get_interped(i, target, ref, invalid_pos_new)
                arr_u[i] = c_u
                arr_v[i] = c_v
            target_interped_u[invalid_pos_new] = arr_u
            target_interped_v[invalid_pos_new] = arr_v
            invalid_interped = utils.generate_mask_invalid(target_interped_u)
            print(f"total invalid values after interpolation with {ref}: {invalid_interped.sum()}")
            print(f"    values filled: {num_invalid_new - invalid_interped.sum()}")
        print(f"total invalid values on interpolated: {invalid_interped.sum()}")

        return target_interped_u, target_interped_v


class SmoothnStep(GapfillStep):
    """
    PLS and smoothing with DCT shenanigans

    uses the matlab engine and smoothn.m
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    https://www.mathworks.com/matlabcentral/fileexchange/25634-smoothn
    """
    def __init__(self, mask=None):
        if mask is not None:
            self.mask = HFRGrid(mask) if not isinstance(mask, HFRGrid) else mask
        else:
            self.mask = None

    def do_validation(self, target):
        if self.mask is None:
            return
        _, targ_lats, targ_lons = target.get_coords()
        targ_min = (targ_lats[0], targ_lons[0])
        targ_max = (targ_lats[-1], targ_lons[-1])
        _, mask_lats, mask_lons = self.mask.get_coords()
        mask_same_res = (len(targ_lats) == len(mask_lats)) and (len(targ_lons) == len(mask_lons))
        if not mask_same_res:
            raise ValueError("Mask is not the same lat/lon shape as target")
        # mask_nc should just be sliced before being used
        # change these asserts to >= later when that's done
        lat_inbounds = (mask_lats[0] == targ_min[0]) and (mask_lats[-1] == targ_max[0])
        lon_inbounds = (mask_lons[0] == targ_min[1]) and (mask_lons[-1] == targ_max[1])
        if not (lat_inbounds and lon_inbounds):
            raise ValueError("Incorrect mask dimensions")

    def process(
        self, u: np.ndarray, v: np.ndarray, target: HFRGrid, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.do_validation(target)

        # DCT smoothing and gapfilling using matlab
        eng = matlab.engine.start_matlab()

        target_smoothed_u = u.copy()
        target_smoothed_v = v.copy()
        u_list = target_smoothed_u.tolist()
        v_list = target_smoothed_v.tolist()

        print(f"Filling {len(u_list)} fields...")
        for i in range(len(u_list)):
            u_mat = matlab.double(u_list[i])
            v_mat = matlab.double(v_list[i])
            uv_smooth = eng.smoothn([u_mat, v_mat], "robust")
            u_array = np.empty(uv_smooth[0].size)
            v_array = np.empty(uv_smooth[1].size)
            u_array[:] = uv_smooth[0]
            v_array[:] = uv_smooth[1]
            target_smoothed_u[i] = u_array
            target_smoothed_v[i] = v_array

        if self.mask is not None:
            no_data = utils.generate_mask_none(self.mask.xrds["U"].values)
            no_data = np.tile(no_data, (target.xrds["time"].size, 1, 1))
            target_smoothed_u[no_data] = np.nan
            target_smoothed_v[no_data] = np.nan

        eng.quit()

        return target_smoothed_u, target_smoothed_v


def import_gapfill_step(name):
    mod = importlib.import_module("src.gapfilling")
    try:
        return getattr(mod, name)
    except AttributeError:
        raise AttributeError(f"Gapfilling step {name} not found in gapfilling.py")


class Gapfiller:
    def __init__(self):
        self.steps = []

    def add_steps(self, *args):
        for step in args:
            if not isinstance(step, GapfillStep):
                raise TypeError(f"{step} is not a proper gapfilling step.")
            self.steps.append(step)

    def execute(self, target: HFRGrid, **kwargs) -> xr.Dataset:
        u = target.xrds["U"].values.copy()
        v = target.xrds["V"].values.copy()
        for step in self.steps:
            u, v = step.process(u, v, target, **kwargs)

        # re-add coordinates, dimensions, and metadata to interpolated data
        darr_u = utils.conv_to_dataarray(u, target.xrds["U"])
        darr_v = utils.conv_to_dataarray(v, target.xrds["V"])
        target_interped_xrds = target.xrds.drop_vars(["U", "V"]).assign(U=darr_u, V=darr_v)
        return target_interped_xrds

    @classmethod
    def load_from_config(cls, *args):
        steps = []
        for step in args:
            step_class = import_gapfill_step(step["name"])
            steps.append(step_class(**step["args"]))
        gapfiller = cls()
        gapfiller.add_steps(*steps)
        return gapfiller
