import logging


logging.basicConfig(
    filename="logs/logs.txt",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG
)
