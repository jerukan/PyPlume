"""
Support and utilities for retrieving datsets servers that have OPENDAP access.
"""
from enum import Enum, unique, auto

import numpy as np
import xarray as xr

import src.utils as utils


# xarray dask chunks, mainly for the stupid big datasets
# CHUNK_SIZE = 50
CHUNK_SIZE = "100MB"


@unique
class ThreddsCode(Enum):
    """Constants for different data sources"""
    USWC_6KM_HOURLY = auto()
    USWC_2KM_HOURLY = auto()
    USWC_1KM_HOURLY = auto()
    USWC_500M_HOURLY = auto()
    USEC_6KM_HOURLY = auto()
    USEC_2KM_HOURLY = auto()
    USEC_1KM_HOURLY = auto()
    DATA_HYCOMFORE = auto()  # hycom forecast data for the east coast, 3 hour intervals


UCSD_HFR_CODES = {
    ThreddsCode.USWC_6KM_HOURLY, ThreddsCode.USWC_2KM_HOURLY, ThreddsCode.USWC_1KM_HOURLY,
    ThreddsCode.USWC_500M_HOURLY, ThreddsCode.USEC_6KM_HOURLY, ThreddsCode.USEC_2KM_HOURLY,
    ThreddsCode.USEC_1KM_HOURLY
}


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


def rename_dataset_vars(path):
    """
    Renames variable/coord keys in an NetCDF ocean current dataset.
    If the data has a depth dimension, it will be removed.

    Args:
        path (path-like or xr.Dataset)
    """
    MAPPINGS = {
        "depth": {"depth", "z"},
        "lat": {"lat", "latitude", "y"},
        "lon": {"lon", "longitude", "long", "x"},
        "time": {"time", "t"},
        "U": {"u", "water_u"},
        "V": {"v", "water_v"}
    }
    if isinstance(path, xr.Dataset):
        ds = path
    else:
        with xr.open_dataset(path) as opened:
            ds = opened
    rename_map = {}
    for var in ds.variables.keys():
        for match in MAPPINGS.keys():
            if var.lower() in MAPPINGS[match]:
                rename_map[var] = match
    ds = ds.rename(rename_map)
    return ds


def drop_depth(ds):
    """
    Depth will still be a coordinate in the whole dataset, but the U and V
    velocity data will not have depth information anymore.
    """
    if "depth" in ds["U"].dims:
        ds["U"] = ds["U"].sel(depth=0)
    if "depth" in ds["V"].dims:
        ds["V"] = ds["V"].sel(depth=0)
    return ds


def preprocess_thredds_dataset(ds, thredds_code):
    """
    Once the data is loaded into memory from some source (internet or local),
    some of the variables/coordinates of the dataset itself might not be
    in the correct format the simulation will read it in.

    Units might be incorrect or the 'latitude' variable might have a different
    name. This method serves to standardize these inconsistencies.
    """
    if thredds_code in UCSD_HFR_CODES:
        return rename_dataset_vars(ds)
    if thredds_code == ThreddsCode.DATA_HYCOMFORE:
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
        ds = ds.sel(depth=0)
        ds = drop_depth(rename_dataset_vars(ds))
        return ds
    if thredds_code == "some other code that needs to be preprocessed...":
        # implement other cases here if needed
        return ...
    # default case, you probably don't want this
    return ds


def retrieve_thredds_dataset(thredds_code):
    """
    If some source needs to be loaded differently than normal, or requires some
    additional processing before actually being sliced and downloaded, define
    that processing here.
    """
    ds = None
    if thredds_code in UCSD_HFR_CODES:
        dropvars = {
            "time_bnds", "depth_bnds", "wgs84", "processing_parameters", "radial_metadata",
            "depth", "time_offset", "dopx", "dopy", "hdop", "number_of_sites",
            "number_of_radials", "time_run"
        }
        ds = xr.open_dataset(
            thredds_urls[thredds_code], chunks={"time": CHUNK_SIZE}, drop_variables=dropvars
        )
    elif thredds_code == ThreddsCode.DATA_HYCOMFORE:
        # HYCOM data times cannot be decoded normally
        dropvars = {"water_temp", "water_temp_bottom", "salinity", "salinity_bottom", "water_u_bottom", "water_v_bottom", "surf_el"}
        ds = xr.open_dataset(
            thredds_urls[thredds_code], chunks={"time": CHUNK_SIZE}, decode_times=False,
            drop_variables=dropvars
        )
    elif thredds_code == "some other code that needs to be loaded differently...":
        # implement other cases here if needed
        ds = ...
    else:
        ds = xr.open_dataset(
            thredds_urls[thredds_code], chunks={"time": CHUNK_SIZE}
        )
    return preprocess_thredds_dataset(ds, thredds_code)


def retrieve_dataset(ds_src):
    """
    Get the full xarray dataset for thredds data at a given thredds dataset

    TODO check if the thredds server is down so it doesn't get stuck
    """
    if ds_src in ThreddsCode.__members__:
        ds_src = ThreddsCode[ds_src]
    # url passed in
    if isinstance(ds_src, str):
        return xr.open_dataset(ds_src, chunks={"time": CHUNK_SIZE})
    if ds_src not in thredds_data or thredds_data[ds_src] is None:
        print(f"Data for type {ds_src} not loaded yet. Loading from...")
        print(thredds_urls[ds_src])
        thredds_data[ds_src] = retrieve_thredds_dataset(ds_src)
    return thredds_data[ds_src]


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
