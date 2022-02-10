"""
Support and utilities for retrieving datsets servers that have OPENDAP access.
"""
from enum import Enum, unique

import numpy as np
import xarray as xr

import src.utils as utils


# xarray dask chunks, mainly for the stupid big datasets
# CHUNK_SIZE = 50
CHUNK_SIZE = "100MB"


@unique
class ThreddsCode(Enum):
    """Constants for different data sources"""
    USWC_6KM_HOURLY = 0
    USWC_2KM_HOURLY = 1
    USWC_1KM_HOURLY = 2
    USWC_500M_HOURLY = 3
    USEC_6KM_HOURLY = 4
    USEC_2KM_HOURLY = 5
    USEC_1KM_HOURLY = 6
    DATA_HYCOMFORE = 12  # hycom forecast data for the east coast, 3 hour intervals


thredds_names = {
    ThreddsCode.USWC_6KM_HOURLY: "US west coast 6km hourly",
    ThreddsCode.USWC_2KM_HOURLY: "US west coast 2km hourly",
    ThreddsCode.USWC_1KM_HOURLY: "US west coast 1km hourly",
    ThreddsCode.USWC_500M_HOURLY: "US west coast 500m hourly",
    ThreddsCode.USEC_6KM_HOURLY: "US east and gulf coast 6km hourly",
    ThreddsCode.USEC_2KM_HOURLY: "US east and gulf coast 2km hourly",
    ThreddsCode.USEC_1KM_HOURLY: "US east and gulf coast 1km hourly",
    ThreddsCode.DATA_HYCOMFORE: "East coast HYCOM forecast",
}


thredds_urls = {
    ThreddsCode.USWC_6KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.USWC_2KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.USWC_1KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.USWC_500M_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/500m/hourly/RTV/HFRADAR_US_West_Coast_500m_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.USEC_6KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/6km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_6km_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.USEC_2KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/2km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_2km_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.USEC_1KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USEGC/1km/hourly/RTV/HFRADAR_US_East_and_Gulf_Coast_1km_Resolution_Hourly_RTV_best.ncd",
    ThreddsCode.DATA_HYCOMFORE: "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/FMRC/GLBy0.08_930_FMRC_best.ncd",
}

# do not access this dict directly, only load datasets when they are needed
# use retrieve_dataset to get this data
# functions like a cache
thredds_data = {}

def retrieve_dataset(thredds_code):
    """
    Get the full xarray dataset for thredds data at a given thredds dataset

    TODO check if the thredds server is down so it doesn't get stuck
    """
    if thredds_code in ThreddsCode.__members__:
        thredds_code = thredds_urls[ThreddsCode[thredds_code]]
    # url passed in
    if isinstance(thredds_code, str):
        return xr.open_dataset(thredds_code, chunks={"time": CHUNK_SIZE})
    if thredds_code not in thredds_data or thredds_data[thredds_code] is None:
        print(f"Data for type {thredds_code} not loaded yet. Loading from...")
        print(thredds_urls[thredds_code])
        thredds_data[thredds_code] = xr.open_dataset(
            thredds_urls[thredds_code], chunks={"time": CHUNK_SIZE}
        )
    return thredds_data[thredds_code]


def get_time_slice(time_range, inclusive=False, ref_coords=None, precision="h"):
    time_range = list(time_range)
    time_range[0] = np.datetime64(time_range[0])
    time_range[1] = np.datetime64(time_range[1])
    if time_range[0] == time_range[1]:
        return slice(np.datetime64(time_range[0], precision),
                     np.datetime64(time_range[1], precision) + np.timedelta64(1, precision))
    if len(time_range) == 2:
        if inclusive:
            if ref_coords is not None:
                time_range = utils.include_coord_range(time_range, ref_coords)
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]))
    if len(time_range) == 3:
        # step size is an integer in hours
        if inclusive:
            interval = time_range[2]
            time_range = utils.include_coord_range(time_range[0:2], ref_coords)
            time_range = (time_range[0], time_range[1], interval)
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]), time_range[2])


def get_region(data):
    time_range = get_time_slice(data[2])
    if data[5]:
        dataset = retrieve_dataset(data[1])
        lat_range = utils.include_coord_range(data[3], dataset["lat"].values)
        lon_range = utils.include_coord_range(data[4], dataset["lon"].values)
    else:
        lat_range = data[3]
        lon_range = data[4]
    return dict(
        name = data[0],
        category = data[1],
        time = time_range,
        lat = lat_range,
        lon = lon_range,
        domain = {
            "S": lat_range[0],
            "N": lat_range[1],
            "W": lon_range[0],
            "E": lon_range[1],
        }
    )

def get_latest_span(delta):
    # GMT, data recorded hourly
    time_now = np.datetime64("now", "h")
    return (time_now - delta, time_now)


def check_bounds(dataset, lat_range, lon_range, time_range):
    times = dataset["time"].values
    lats = dataset["lat"].values
    lons = dataset["lon"].values
    span_checker = lambda rng, coords: rng[0] <= coords[0] or rng[1] >= coords[-1]
    if span_checker(time_range, times):
        print("Timespan reaches the min/max of the range")
    if span_checker(lat_range, lats):
        print("Latitude span reaches the min/max of the range")
    if span_checker(lon_range, lons):
        print("Longitude span reaches the min/max of the range")


def get_thredds_dataset(thredds_code, time_range, lat_range, lon_range,
        inclusive=False, padding=0.0) -> xr.Dataset:
    """
    Params:
        thredds_code (int or str): the dataset constant or url
        time_range (np.datetime64, np.datetime64[, int]): (start, stop[, interval])
        lat_range (float, float)
        lon_range (float, float)
        inclusive (bool)
        padding (float): lat and lon padding

    Returns:
        xr.Dataset
    """
    print("Retrieving thredds dataset...")
    reg_data = retrieve_dataset(thredds_code)
    if inclusive:
        lat_range = utils.include_coord_range(lat_range, reg_data["lat"].values)
        lon_range = utils.include_coord_range(lon_range, reg_data["lon"].values)
    lat_range = (lat_range[0] - padding, lat_range[1] + padding)
    lon_range = (lon_range[0] - padding, lon_range[1] + padding)
    if not isinstance(time_range, slice):
        if time_range[0] == "START":
            time_range = (reg_data["time"].values[0], time_range[1])
        if time_range[1] == "END":
            time_range = (time_range[0], reg_data["time"].values[-1])
        time_slice = get_time_slice(time_range, inclusive=inclusive, ref_coords=reg_data["time"].values)
    else:
        time_slice = time_range
    check_bounds(reg_data, lat_range, lon_range, (time_slice.start, time_slice.stop))
    dataset_start = reg_data["time"].values[0]
    if time_slice.start >= np.datetime64("now") or time_slice.stop <= dataset_start:
        raise ValueError("Desired time range is out of range for the dataset")
    sliced_data = reg_data.sel(
        time=time_slice,
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    if len(sliced_data["time"]) == 0:
        raise ValueError("No timestamps inside given time interval")
    return sliced_data
