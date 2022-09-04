"""
Support and utilities for retrieving datsets from thredds servers that have OPeNDAP (preferably)
access.
"""
import logging

import numpy as np
import xarray as xr

from pyplume import get_logger
from pyplume.dataloaders import *


logger = get_logger(__name__)

VAR_MAPPINGS_HFRNET_UCSD = {
    "U": {"u"},
    "V": {"v"}
}
DROP_VARS_HFRNET_UCSD = {
    "time_bnds", "depth_bnds", "wgs84", "processing_parameters", "radial_metadata",
    "depth", "time_offset", "dopx", "dopy", "hdop", "number_of_sites",
    "number_of_radials", "time_run"
}
VAR_MAPPINGS_FMRC_HYCOM = {
    "U": {"water_u"},
    "V": {"water_v"}
}
DROP_VARS_FMRC_HYCOM = {
    "water_temp", "water_temp_bottom", "salinity", "salinity_bottom", "water_u_bottom",
    "water_v_bottom", "surf_el"
}


def get_hfrnet_ucsd_load_method(time_chunk_size=None):
    def new_load_method(src):
        return drop_depth(
            get_simple_load_method(
                mappings=VAR_MAPPINGS_HFRNET_UCSD,
                drop_vars=DROP_VARS_HFRNET_UCSD,
                time_chunk_size=time_chunk_size
            )(src)
        )
    return new_load_method


def get_fmrc_hycom_load_method(time_chunk_size=None):
    time_chunks = parse_time_chunk_size(time_chunk_size)
    def load_hycom_method(src):
        # HYCOM data times cannot be decoded normally
        ds = open_dataset(
            src, chunks=time_chunks, drop_variables=DROP_VARS_FMRC_HYCOM, decode_times=False
        )
        # This particular HYCOM forecast data has different units of time, where
        # it is "hours since <time from a week ago> UTC", which has to be converted
        # to propert datetime values
        # hacky way of getting the time origin of the data
        t0 = np.datetime64(ds.time.units[12:35])
        tmp = ds["time"].data
        ds["t0"] = np.timedelta64(t0 - np.datetime64("0000-01-01T00:00:00.000"), "h") / np.timedelta64(1, "D")
        # replace time coordinate data with actual datetimes
        ds = ds.assign_coords(time=(t0 + np.array(tmp, dtype="timedelta64[h]")))
        # modify metadata
        ds["time"].attrs["long_name"] = "Forecast time"
        ds["time"].attrs["standard_name"] = "time"
        ds["time"].attrs["_CoordinateAxisType"] = "Time"
        ds["tau"].attrs["units"] = "hours since " + ds["tau"].time_origin
        # drop depth data
        ds = drop_depth(rename_dataset_vars(ds, VAR_MAPPINGS_FMRC_HYCOM))
        return ds
    return load_hycom_method


SRC_THREDDS_HFRNET_UCSD = DataSource(
    id="THREDDS_HFRNET_UCSD",
    name="HFRnet Thredds Data Server",
    available_datasets=[
        DatasetInfo(
            id="USWC_6KM_HOURLY",
            name="US west coast 6km hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd"
        ),
        DatasetInfo(
            id="USWC_2KM_HOURLY",
            name="US west coast 2km hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd"
        ),
        DatasetInfo(
            id="USWC_1KM_HOURLY",
            name="US west coast 1km hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"
        ),
        DatasetInfo(
            id="USWC_500M_HOURLY",
            name="US west coast 500m hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/500m/hourly/RTV/HFRADAR_US_West_Coast_500m_Resolution_Hourly_RTV_best.ncd"
        ),
        DatasetInfo(
            id="USEC_6KM_HOURLY",
            name="US east and gulf coast 6km hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/6km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_6km_Resolution_Hourly_RTV_best.ncd"
        ),
        DatasetInfo(
            id="USEC_2KM_HOURLY",
            name="US east and gulf coast 2km hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/2km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_2km_Resolution_Hourly_RTV_best.ncd"
        ),
        DatasetInfo(
            id="USEC_1KM_HOURLY",
            name="US east and gulf coast 1km hourly",
            url="http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/1km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_1km_Resolution_Hourly_RTV_best.ncd"
        )
    ],
    load_method=get_hfrnet_ucsd_load_method(time_chunk_size=CHUNK_SIZE_DEFAULT)
)
SRC_THREDDS_HYCOM = DataSource(
    id="THREDDS_HYCOM",
    name="HYCOM Thredds Data Server",
    available_datasets=[
        DatasetInfo(
            id="FMRC_HYCOM",
            name="HYCOM + NCODA Global 1/12 Analysis FMRC",
            url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/GLBy0.08_930_FMRC_best.ncd",
        )
    ],
    load_method=get_fmrc_hycom_load_method(time_chunk_size=CHUNK_SIZE_DEFAULT)
)


AVAILABLE_SRCS = [
    SRC_THREDDS_HFRNET_UCSD, SRC_THREDDS_HYCOM
]
AVAILABLE_SRCS_MAP = {}
for src in AVAILABLE_SRCS:
    AVAILABLE_SRCS_MAP[src.id] = src


def register_data_source(ds_src):
    if ds_src.id in AVAILABLE_SRCS_MAP:
        logger.info(f"{ds_src.id} is already registered, overriding data source")
    AVAILABLE_SRCS.add(ds_src)
    AVAILABLE_SRCS_MAP[ds_src.id] = ds_src


def retrieve_dataset(src_id, ds_id):
    """
    Get the full xarray dataset for thredds data at a given thredds dataset

    TODO check if the thredds server is down so it doesn't get stuck
    """
    if src_id in AVAILABLE_SRCS_MAP.keys():
        ds_src = AVAILABLE_SRCS_MAP[src_id]
    else:
        raise ValueError(f"{src_id} is not registered")
    return ds_src.load_source(ds_id)
