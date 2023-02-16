from abc import ABC, abstractmethod
import importlib
import logging
import os
import sys
from typing import Tuple

import numpy as np
import xarray as xr

from pyplume import get_logger
from pyplume.dataloaders import slice_dataset, SurfaceGrid, DataLoader
from pyplume.gapfill_algs import dctpls, eof_functions
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


class LowResOversample(GapfillStep):
    """
    Oversamples data from lower resolution data in order to fill gaps in higher resolution
    data.
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
            lat_inbounds = (ref_lats[0] <= targ_min[0]) and (
                ref_lats[-1] >= targ_max[0]
            )
            lon_inbounds = (ref_lons[0] <= targ_min[1]) and (
                ref_lons[-1] >= targ_max[1]
            )
            time_inbounds = (ref_times[0] <= targ_times[0]) and (
                ref_times[-1] >= targ_times[-1]
            )
            if not (lat_inbounds and lon_inbounds and time_inbounds):
                raise ValueError(
                    "Incorrect reference dimensions (reference dimension ranges \
                    should be larger than the target's)"
                )

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
                loaded_references.append(
                    SurfaceGrid(
                        # slice the data before loading into SurfaceGrid since it's huge
                        DataLoader(
                            ref,
                            time_range=time_range,
                            lat_range=lat_range,
                            lon_range=lon_range,
                            inclusive=True,
                        ).dataset
                    )
                )
            elif isinstance(ref, str):
                # TODO generalize this
                # slice the data before loading into SurfaceGrid since it's huge
                ref = DataLoader(
                    ref,
                    datasource=thredds_data.SRC_THREDDS_HFRNET_UCSD,
                    time_range=time_range,
                    lat_range=lat_range,
                    lon_range=lon_range,
                    inclusive=True,
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


class DCTPLS(GapfillStep):
    """
    PLS and smoothing with DCT shenanigans

    Based off smoothn.m from MATLAB.
    https://www.mathworks.com/matlabcentral/fileexchange/25634-smoothn

    REFERENCES (please refer to the two following papers)
    ---------
    1) Garcia D, Robust smoothing of gridded data in one and higher
    dimensions with missing values. Computational Statistics & Data
    Analysis, 2010;54:1167-1178.
    http://www.biomecardio.com/publis/csda10.pdf
    2) Garcia D, A fast all-in-one method for automated post-processing of
    PIV data. Exp Fluids, 2011;50:1247-1259.
    http://www.biomecardio.com/publis/expfluids11.pdf

    Written by Damien Garcia
    """

    def __init__(self, exclude_oob=True, **smoothn_kwargs):
        """
        Args:
            exclude_oob (bool): If True, exclude values outside the intended data domain.
            **smoothn_kwargs: Keyword arguments for running the smoothn function from dctpls.py.
        """
        self.exclude_oob = exclude_oob
        self.smoothn_kwargs = {} if smoothn_kwargs is None else smoothn_kwargs
        # prevent return errors
        self.smoothn_kwargs["full_output"] = False

    def process(
        self, u: np.ndarray, v: np.ndarray, target: xr.Dataset, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Filling {len(u)} fields...")
        u_smooth, v_smooth = dctpls.smoothn(
            u, v, **self.smoothn_kwargs
        )
        target_smoothed_u = u_smooth
        target_smoothed_v = v_smooth

        if self.exclude_oob:
            nodata_mask = utils.generate_mask_no_data(target["U"], tile=True)
            target_smoothed_u[nodata_mask] = np.nan
            target_smoothed_v[nodata_mask] = np.nan

        return target_smoothed_u, target_smoothed_v


class DINEOF(GapfillStep):
    """
    DINEOF spatial gapfilling.
    """

    def __init__(self, exclude_oob=True, modemax=None, maxits=None, thresh=None):
        """
        Args:
            exclude_oob (bool): If True, exclude values outside the intended data domain.
        """
        self.exclude_oob = exclude_oob
        self.modemax = 10 if modemax is None else modemax
        self.maxits = 10 if maxits is None else maxits
        self.thresh = 0.05 if thresh is None else thresh

    def process(
        self, u: np.ndarray, v: np.ndarray, target: xr.Dataset, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        t, latsz, lonsz = u.shape
        umask = u.reshape((t, latsz * lonsz))
        umask = np.ma.array(umask, mask=np.isnan(umask))
        vmask = v.reshape((t, latsz * lonsz))
        vmask = np.ma.array(vmask, mask=np.isnan(vmask))
        ufilled, _ = eof_functions.fill_gappy_EOF(umask, self.modemax, self.maxits, self.thresh)
        vfilled, _ = eof_functions.fill_gappy_EOF(vmask, self.modemax, self.maxits, self.thresh)
        ufilled = ufilled.reshape((t, latsz, lonsz))
        vfilled = vfilled.reshape((t, latsz, lonsz))
        if self.exclude_oob:
            nodata_mask = utils.generate_mask_no_data(target["U"], tile=True)
            ufilled[nodata_mask] = np.nan
            vfilled[nodata_mask] = np.nan
        return ufilled, vfilled


class Gapfiller:
    def __init__(self, *args):
        self.steps = []
        self.add_steps(*args)

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
