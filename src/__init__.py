import logging.config
import os
import yaml

from . import data
from . import experiments
from . import methods
from . import util

__all__ = ['data', 'experiments', 'methods', 'util']
__author__ = 'Otilia Stretcu'

__logging_config_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
if os.path.exists(__logging_config_path):
    with open(__logging_config_path, 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
else:
    logging.getLogger('').addHandler(logging.NullHandler())
