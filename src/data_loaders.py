import numpy as np
import xarray as xr


def rename_dataset_vars(src, mappings):
    """
    Renames variable/coord keys in an NetCDF ocean current dataset.

    Args:
        src (path-like or xr.Dataset)
        mappings (dict): format:
            {
                "standardized_var_name": {"other", "possible", "names"},
                ...
            }
    """
    if isinstance(src, xr.Dataset):
        ds = src
    else:
        with xr.open_dataset(src) as opened:
            ds = opened
    rename_map = {}
    for var in ds.variables.keys():
        for match in mappings.keys():
            if var.lower() in mappings[match]:
                rename_map[var] = match
    ds = ds.rename(rename_map)
    return ds


def drop_depth(ds):
    """
    Depth will still be a coordinate in the whole dataset, but the U and V
    velocity data will not have depth information anymore.

    Args:
        ds (xr.Dataset): standardized vector field
    """
    if "depth" in ds["U"].dims:
        ds["U"] = ds["U"].sel(depth=0)
    if "depth" in ds["V"].dims:
        ds["V"] = ds["V"].sel(depth=0)
    return ds


class VecDataLoader:
    """
    An abstract class which defines a way to load data from some source into a vector
    field (as an xarray dataset). You can define any sort of configuration needed on
    a per-source basis by extending this class.
    """

    def load_source(self, src):
        """
        Loads a dataset from some source, and processes it so it is a standard format
        for the simulation to read.

        Args:
            src (str or path-like): a url or path to the data

        Returns:
            xr.Dataset: a dataset in the standardized format
        """
        return xr.open_dataset(src)


class SimpleDataLoader(VecDataLoader):
    DEFAULT_MAPPINGS = {
        "depth": {"depth", "z"},
        "lat": {"lat", "latitude", "y"},
        "lon": {"lon", "longitude", "long", "x"},
        "time": {"time", "t"},
        "U": {"u", "water_u"},
        "V": {"v", "water_v"}
    }

    def __init__(self, mappings=None, dropvars=None, time_chunk_size=None):
        if mappings is None:
            self.mappings = self.DEFAULT_MAPPINGS
        else:
            self.mappings = mappings
        self.dropvars = dropvars
        if time_chunk_size is not None:
            self.time_chunks = {"time": time_chunk_size}
        else:
            self.time_chunks = None
    
    def load_source(self, src):
        ds = xr.open_dataset(src, chunks=self.time_chunks, drop_variables=self.dropvars)
        if self.mappings is None:
            return ds
        return drop_depth(rename_dataset_vars(ds, self.mappings))


class HFRThreddsDataLoader(SimpleDataLoader):
    MAPPINGS = {
        "U": {"u"},
        "V": {"v"}
    }
    DROPVARS = {
        "time_bnds", "depth_bnds", "wgs84", "processing_parameters", "radial_metadata",
        "depth", "time_offset", "dopx", "dopy", "hdop", "number_of_sites",
        "number_of_radials", "time_run"
    }
    CHUNK_SIZE = "100MB"

    def __init__(self):
        super().__init__(self.MAPPINGS, self.DROPVARS, self.CHUNK_SIZE)


class HYCOMDataLoader(VecDataLoader):
    MAPPINGS = {
        "U": {"water_u"},
        "V": {"water_v"}
    }
    DROPVARS = {
        "water_temp", "water_temp_bottom", "salinity", "salinity_bottom", "water_u_bottom",
        "water_v_bottom", "surf_el"
    }
    CHUNK_SIZE = "100MB"

    def load_source(self, src):
        # HYCOM data times cannot be decoded normally
        ds = xr.open_dataset(
            src, chunks=self.CHUNK_SIZE, drop_variables=self.DROPVARS, decode_times=False
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
        ds = ds.sel(depth=0)
        ds = drop_depth(rename_dataset_vars(ds, self.MAPPINGS))
        return ds
