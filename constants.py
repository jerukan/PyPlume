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
    N=32.585,
    W=-117.162,
    E=-117.105
)

TIJUANA_MOUTH_POSITION = np.array([32.5567724355310, -117.130164948310])
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

SD_FULL_COASTLINE_FILENAME = "coastOR2Mex.mat"
SD_FULL_TIJUANA_IDXS = np.array([3692, 3693, 3694, 3695, 3696, 3697, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3706, 3707, 3708, 3709, 3710, 3711, 3712, 3713, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 4767])
