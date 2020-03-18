"""
"""

import numpy as np
from polyglot.detect import Detector

### Defining Language detection
def detect_lang_polyglot(text, simplify = True, print_error = False, raise_error = False, keep_unreliable = False, **kwargs):

  """
  For detecting language using polyglot, but with exception handling

  Examples:
  >>> detect_lang("This is a test text")
  ('en', 95.0)
  >>> detect_lang(text = "Dette er åbenbart en norsk tekst", keep_unreliable = True)
  ('no', 97.0)
  >>> detect_lang(text = "Dette er åbenbart en norsk tekst. This is also an english text.", keep_unreliable = True)
  """
  text = str(text)
  try:
    detector = Detector(text, quiet=True)
    if detector.reliable or keep_unreliable:
      lang = detector.language
      if simplify:
        return lang.code
      return lang.code, lang.confidence
  except Exception as e:
    if print_error and not raise_error:
      print(e)
    if raise_error:
      raise Exception(e) 
  if simplify:
    return np.nan
  return np.nan, np.nan
