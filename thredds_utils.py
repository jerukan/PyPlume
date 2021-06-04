"""
TODO add support to get regions other than just US west coast
"""

import numpy as np
import xarray as xr

import utils

USWC_6KM_HOURLY = 0
USWC_2KM_HOURLY = 1
USWC_1KM_HOURLY = 2
USWC_500M_HOURLY = 3

num_chunks = 50

thredds_urls = {
    USWC_6KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd",
    USWC_2KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd",
    USWC_1KM_HOURLY: "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"
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
    if thredds_code not in thredds_data or thredds_data[thredds_code] is None:
        print(f"Data for type {thredds_code} not loaded yet. Loading from...")
        print(thredds_urls[thredds_code])
        thredds_data[thredds_code] = xr.open_dataset(
            thredds_urls[thredds_code], chunks={"time": num_chunks}
        )
    return thredds_data[thredds_code]


def get_time_slice(time_range, inclusive=False, ref_coords=None, precision="h"):
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
        print("Time spans entire dataset")
    if span_checker(lat_range, lats):
        print("Latitude spans its entire range")
    if span_checker(lon_range, lons):
        print("Longitude spans its entire range")


def get_thredds_dataset(thredds_code, time_range, lat_range, lon_range,
        inclusive=False, padding=0.0) -> xr.Dataset:
    """
    Params:
        name (str)
        thredds_code (int)
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
    time_slice = get_time_slice(time_range, inclusive=inclusive, ref_coords=reg_data["time"].values)
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
