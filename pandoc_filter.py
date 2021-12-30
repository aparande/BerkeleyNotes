#!/usr/bin/env python3

import yaml

from latex_filters.filters import *
from latex_filters import PandocState

from pandocfilters import toJSONFilters

if __name__ == '__main__':
  with open("pandoc.yaml", 'r') as f:
    config = yaml.safe_load(f)

  state = PandocState()
  filters = [eval(conf['class_name'])(conf['config'], state) for conf in config]

  toJSONFilters(filters)

