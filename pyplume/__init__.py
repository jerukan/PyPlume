import logging
import sys


logging.basicConfig(
    filename="logs/output.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    # call default excepthook
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    if issubclass(exc_type, KeyboardInterrupt):
        return
    # create a critical level log message with info from the except hook.
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_unhandled_exception
