import sys
import re

from typing import NamedTuple, List, Any
from pandocfilters import Math, Str, RawInline, Para, Header

from latex_filters.filters import CustomNumberedEnvFilter

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

