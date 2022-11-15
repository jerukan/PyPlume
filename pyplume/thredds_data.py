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
    "water_v_bottom", "surf_el", "tau"
}


class HFRNetUCSDLoad:
    def __init__(self, time_chunk_size=None):
        self.time_chunk_size = time_chunk_size
    
    def __call__(self, src):
        return drop_depth(
            SimpleLoad(
                mappings=VAR_MAPPINGS_HFRNET_UCSD,
                drop_vars=DROP_VARS_HFRNET_UCSD,
                time_chunk_size=self.time_chunk_size
            )(src)
        )


class HYCOMLoad:
    def __init__(self, time_chunk_size=None):
        self.time_chunk_size = time_chunk_size
    
    def __call__(self, src):
        time_chunks = parse_time_chunk_size(self.time_chunk_size)
        # HYCOM data times cannot be decoded normally
        ds = xr.open_dataset(
            src, chunks=time_chunks, drop_variables=DROP_VARS_FMRC_HYCOM
        )
        # drop depth data
        ds = drop_depth(rename_dataset_vars(ds, VAR_MAPPINGS_FMRC_HYCOM))
        # change range of longitude values from 0-360 to -180-180
        # ds["lon"] = (ds["lon"] + 180) % 360 - 180
        # ds = ds.sortby("lon")
        return ds


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
    load_method=HFRNetUCSDLoad(time_chunk_size=CHUNK_SIZE_DEFAULT)
)
SRC_THREDDS_HYCOM = DataSource(
    id="THREDDS_HYCOM",
    name="HYCOM Thredds Data Server",
    available_datasets=[
        DatasetInfo(
            id="GLOBAL_FMRC",
            name="HYCOM + NCODA Global 1/12 Analysis FMRC",
            url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/GLBy0.08_930_FMRC_best.ncd",
        ),
        DatasetInfo(
            id="GLOBAL_HINDCAST",
            name="HYCOM + NCODA Global 1/12 Analysis Hindcast Data: 3-hourly",
            url="https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z",
        )
    ],
    load_method=HYCOMLoad(time_chunk_size=CHUNK_SIZE_DEFAULT)
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


def retrieve_dataloader(src_id, ds_id, **kwargs):
    if src_id in AVAILABLE_SRCS_MAP.keys():
        ds_src = AVAILABLE_SRCS_MAP[src_id]
    else:
        raise ValueError(f"{src_id} is not registered")
    return DataLoader(ds_id, datasource=ds_src, **kwargs)


def retrieve_dataset(src_id, ds_id):
    """
    Get the full xarray dataset for thredds data at a given thredds dataset

    TODO check if the thredds server is down so it doesn't get stuck
    """
    return retrieve_dataloader(src_id, ds_id).dataset
