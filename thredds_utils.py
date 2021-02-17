import numpy as np
import xarray as xr

import utils


dataset_url_6kmhourly = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/6km/hourly/RTV/HFRADAR_US_West_Coast_6km_Resolution_Hourly_RTV_best.ncd"
dataset_url_2kmhourly = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd"
dataset_url_1kmhourly = "http://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/1km/hourly/RTV/HFRADAR_US_West_Coast_1km_Resolution_Hourly_RTV_best.ncd"

num_chunks = 50

thredds_data = {
    utils.DATA_6KM: xr.open_dataset(dataset_url_6kmhourly, chunks={"time": num_chunks}),
    utils.DATA_2KM: xr.open_dataset(dataset_url_2kmhourly, chunks={"time": num_chunks}),
    utils.DATA_1KM: xr.open_dataset(dataset_url_1kmhourly, chunks={"time": num_chunks})
}

def get_region(data):
    time_range = get_time_slice(data[2])
    if data[5]:
        lat_range = utils.expand_coord_rng(data[3], thredds_data[data[1]]["lat"].values)
        lon_range = utils.expand_coord_rng(data[4], thredds_data[data[1]]["lon"].values)
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


def get_time_slice(time_range):
    if len(time_range) == 2:
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]))
    if len(time_range) == 3:
        # step size is an integer in hours
        return slice(np.datetime64(time_range[0]), np.datetime64(time_range[1]), time_range[2])


def get_thredds_dataset(name, resolution, time_range, lat_range, lon_range,
        expand_coords=False) -> xr.Dataset:
    """
    Params:
        name (str)
        resolution (int)
        time_range (np.datetime64, np.datetime64[, int]): (start, stop[, interval])
        lat_range (float, float)
        lon_range (float, float)
        expand_coords (bool)

    Returns:
        xr.Dataset
    """
    reg_data = thredds_data[resolution]
    if expand_coords:
        lat_range = utils.expand_coord_rng(lat_range, reg_data["lat"].values)
        lon_range = utils.expand_coord_rng(lon_range, reg_data["lon"].values)
    time_slice = get_time_slice(time_range)
    dataset_start = reg_data["time"].values[0]
    if time_slice.start >= np.datetime64("now") or time_slice.stop <= dataset_start:
        raise ValueError(f"Desired time range is out of range for the dataset")
    sliced_data = reg_data.sel(
        time=time_slice,
        lat=slice(lat_range[0], lat_range[1]),
        lon=slice(lon_range[0], lon_range[1]),
    )
    return sliced_data
