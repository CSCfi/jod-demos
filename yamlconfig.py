import yaml
from yaml.loader import SafeLoader
import os

CONFIGFILE = "config.yml"

def read_config(verbose=False):
    assert os.path.isfile(CONFIGFILE), "Config file missing"
    with open(CONFIGFILE) as cfg:
          config = yaml.load(cfg, Loader=SafeLoader)
    if verbose:
        print('Config file contents:')
        for c in config:
            print(c, end=" ")
            print()
    return config
