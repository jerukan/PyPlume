"""Useful constants"""
import numpy as np

TIJUANA_RIVER_DOMAIN = dict(
    S=32.525,
    N=32.7,
    W=-117.27,
    E=-117.09
)

TIJUANA_MOUTH_DOMAIN = dict(
    S=32.53,
    N=32.564,
    W=-117.162,
    E=-117.105
)

SD_COASTLINE_FILENAME = "coastline.mat"
SD_STATION_FILENAME = "wq_stposition.mat"
SD_STATION_NAMES = np.array([
    "Coronado (North Island)",
    "Silver Strand",
    "Silver Strand Beach",
    "Carnation Ave.",
    "Imperial Beach Pier",
    "Cortez Ave.",
    "End of Seacoast Dr.",
    "3/4 mi. N. of TJ River Mouth",
    "Tijuana River Mouth",
    "Monument Rd.",
    "Board Fence",
    "Mexico"
])
