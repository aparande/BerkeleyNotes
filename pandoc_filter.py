#!/usr/bin/env python3

import yaml
import sys
import re
from typing import NamedTuple, List, Any

from latex_filters.filters import *
from latex_filters import PandocState

from pandocfilters import toJSONFilters
from pandocfilters import Math, Str, RawInline, Para, Header

class NumberedEnvHintFilter(CustomNumberedEnvFilter):
  """
  Convert a custom numbered environment (created with \\newtheorem in LaTeX) but
  add hint tags
  """

  def custom_numbered_env(self, code:str, env):
    """
    Process a custom numbered environment 
    """

    header, output = super().custom_numbered_env(code, env)

    hint_tag = Para([Str("{% hint style=\"info\" %}")])
    end_hint_tag = Para([Str("{% endhint %}")])
    return [hint_tag, header, output, end_hint_tag]

class TextDegreeFilter(PandocFilter):
  def __call__(self, key:str, value:Any, fmt:str, meta:Any):
    if key == "RawInline":
      [fmt, code] = value
      if fmt == "latex" and re.match("\\\\textdegree", code):
        return Str("˚")

if __name__ == '__main__':
  with open("pandoc.yaml", 'r') as f:
    config = yaml.safe_load(f)

  state = PandocState()
  filters = [eval(conf['class_name'])(conf['config'], state) for conf in config]

  toJSONFilters(filters)

