import logging

from datapizza.core.utils import _basic_config

# Setup base logging

# Create and configure the main package logger
log = logging.getLogger("datapizza")
_basic_config(log)

log.setLevel(logging.DEBUG)
