import logging
import sys
import warnings

import numpy as np

# ignore common deprecation warnings that pop up constantly
warnings.simplefilter("ignore", UserWarning)
# ignore divide by nan error that happens constantly with parcels
np.seterr(divide="ignore", invalid="ignore")


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler("logs/output.log")
    handler.setFormatter(
        logging.Formatter("%(asctime)s,%(msecs)d %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger

# logging.basicConfig(
#     filename="logs/output.log",
#     filemode="a",
#     format="%(asctime)s,%(msecs)d %(name)s | %(levelname)s | %(message)s",
#     datefmt="%H:%M:%S",
#     level=logging.DEBUG
# )
logger = get_logger(__name__)


def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    # call default excepthook
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    if issubclass(exc_type, KeyboardInterrupt):
        return
    # create a critical level log message with info from the except hook.
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_unhandled_exception
