from abc import ABC, abstractmethod
import importlib
import logging
import os
import sys
from typing import Tuple

import matlab.engine
import numpy as np
import xarray as xr

from pyplume import get_logger
from pyplume.dataloaders import slice_dataset, SurfaceGrid, DataLoader
import pyplume.utils as utils
import pyplume.thredds_data as thredds_data


logger = get_logger(__name__)


class GapfillStep(ABC):
    @abstractmethod
    def process(
        self, u: np.ndarray, v: np.ndarray, target: xr.Dataset, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


def get_interped(i, target, ref, invalid_where):
    """
    Args:
        i (int): index on invalid_where
        ref (SurfaceGrid): reference Dataset
        invalid_where (array-like): (3, n) dimensional array representing all invalid positions
    
    Returns:
        (u, v): (nan, nan) if no data was found, interpolated values otherwise
    """
    time_diff = np.diff(ref.fieldset_flat.U.grid.time)[0]
    t = invalid_where[0][i]
    lat = target.lats[invalid_where[1][i]]
    lon = target.lons[invalid_where[2][i]]
    current_u, current_v = ref.get_fs_vector(t * time_diff, lat, lon)
    current_abs = abs(current_u) + abs(current_v)
    # if both the u and v components are 0, there's probably no data there
    if np.isnan(ref.get_closest_current(t, lat, lon)[0]) or current_abs == 0:
        return np.nan, np.nan
    return current_u, current_v


class InterpolationStep(GapfillStep):
    """
    Uses linear interpolation
    """
    def __init__(self, references):
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
        self, u: np.ndarray, v: np.ndarray, target: xr.Dataset, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        target = SurfaceGrid(target, init_fs=False)
        times, lats, lons = target.get_coords()
        time_range = (times[0], times[-1])
        lat_range = (lats[0], lats[-1])
        lon_range = (lons[0], lons[-1])
        loaded_references = []
        for i, ref in enumerate(self.references):
            logger.info(f"Loading interp reference {ref}")
            if isinstance(ref, SurfaceGrid):
                loaded_references.append(ref)
            elif isinstance(ref, xr.Dataset):
                loaded_references.append(SurfaceGrid(
                    # slice the data before loading into SurfaceGrid since it's huge
                    DataLoader(
                        ref, time_range=time_range, lat_range=lat_range, lon_range=lon_range,
                        inclusive=True
                    ).dataset
                ))
            elif isinstance(ref, str):
                # TODO generalize this
                # slice the data before loading into SurfaceGrid since it's huge
                ref = DataLoader(
                    ref, datasource=thredds_data.SRC_THREDDS_HFRNET_UCSD, time_range=time_range,
                    lat_range=lat_range, lon_range=lon_range, inclusive=True
                ).dataset
                loaded_references.append(SurfaceGrid(ref))
            else:
                raise TypeError(f"Unrecognized type for {ref}")
                        
        self.do_validation(target, loaded_references)
        invalid = utils.generate_mask_invalid(u)
        num_invalid = invalid.sum()
        logger.info(f"total invalid values on target data: {num_invalid}")

        # linear interpolation from lower resolution data
        target_interped_u = u.copy()
        target_interped_v = v.copy()
        invalid_interped = invalid.copy()
        for ref in loaded_references:
            invalid_pos_new = np.where(invalid_interped)
            num_invalid_new = int(invalid_interped.sum())
            arr_u = np.zeros(num_invalid_new)
            arr_v = np.zeros(num_invalid_new)
            logger.info(f"Attempting to interpolate {num_invalid_new} points...")
            for i in range(num_invalid_new):
                c_u, c_v = get_interped(i, target, ref, invalid_pos_new)
                arr_u[i] = c_u
                arr_v[i] = c_v
            target_interped_u[invalid_pos_new] = arr_u
            target_interped_v[invalid_pos_new] = arr_v
            invalid_interped = utils.generate_mask_invalid(target_interped_u)
            logger.info(
                f"total invalid values after interpolation with {ref}: {invalid_interped.sum()}"
                + f"\n\tvalues filled: {num_invalid_new - invalid_interped.sum()}"
            )
        logger.info(f"total invalid values on interpolated: {invalid_interped.sum()}")

        return target_interped_u, target_interped_v


class SmoothnStep(GapfillStep):
    """
    PLS and smoothing with DCT shenanigans

    uses the matlab engine and smoothn.m
    https://www.mathworks.com/help/matlab/matlab-engine-for-python.html
    https://www.mathworks.com/matlabcentral/fileexchange/25634-smoothn
    """
    def __init__(self, mask=None):
        """
        Args:
            mask: (lat, lon): True where should exist, False otherwise
        """
        if mask is None:
            self.mask = None
            return
        if isinstance(mask, str):
            self.mask = np.load(mask)
        else:
            # if we leave this as a DataArray, it could internally store as a dask array, which for
            # some reason hangs indefinitely on some array operations.
            self.mask = np.array(mask)
        if len(self.mask.shape) != 2:
            raise ValueError(f"Incorrect number of mask dimensions ({len(self.mask.shape)})")

    def do_validation(self, target):
        if self.mask is None:
            return
        _, targ_lats, targ_lons = target.get_coords()
        mask_same_res = (len(targ_lats) == len(self.mask)) and (len(targ_lons) == len(self.mask[0]))
        if not mask_same_res:
            raise ValueError(f"Mask (shape {(len(self.mask), len(self.mask[0]))}) is not the same lat/lon shape as target (shape {(len(targ_lats), len(targ_lons))})")

    def process(
        self, u: np.ndarray, v: np.ndarray, target: xr.Dataset, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        target = SurfaceGrid(target, init_fs=False)
        self.do_validation(target)

        # DCT smoothing and gapfilling using matlab
        eng = matlab.engine.start_matlab()

        target_smoothed_u = u.copy()
        target_smoothed_v = v.copy()
        u_list = target_smoothed_u.tolist()
        v_list = target_smoothed_v.tolist()

        logger.info(f"Filling {len(u_list)} fields...")
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
            no_data = np.tile(~self.mask, (target.ds["time"].size, 1, 1))
            target_smoothed_u[no_data] = np.nan
            target_smoothed_v[no_data] = np.nan

        eng.quit()

        return target_smoothed_u, target_smoothed_v


class Gapfiller:
    def __init__(self, *args):
        self.steps = list(args)

    def add_steps(self, *args):
        for step in args:
            if not isinstance(step, GapfillStep):
                raise TypeError(f"{step} is not a proper gapfilling step.")
            self.steps.append(step)

    def execute(self, target: xr.Dataset, **kwargs) -> xr.Dataset:
        if not self.steps:
            return target
        logger.info(f"Executing gapfiller on target {target} with steps {self.steps}")
        u = target["U"].values.copy()
        v = target["V"].values.copy()
        for step in self.steps:
            logger.info(f"Executing step {step}")
            u, v = step.process(u, v, target, **kwargs)

        # re-add coordinates, dimensions, and metadata to interpolated data
        darr_u = utils.conv_to_dataarray(u, target["U"])
        darr_v = utils.conv_to_dataarray(v, target["V"])
        target_interped = target.drop_vars(["U", "V"]).assign(U=darr_u, V=darr_v)
        logger.info(f"Completed gapfilling on target {target}")
        return target_interped

    @classmethod
    def load_from_config(cls, *args):
        steps = []
        for step in args:
            step_class = utils.import_attr(step["path"])
            steps.append(step_class(**step["args"]))
        gapfiller = cls()
        gapfiller.add_steps(*steps)
        return gapfiller
