"""
"""
import os
import sys
import contextlib

import numpy as np

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
  from polyglot.detect import Detector

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

def silence(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sav = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sav
        # pass the return value of the method back
        return value

    return func_wrapper


def get_test_data():
    with open("test_data/fyrtårnet.txt", "r") as f:
        text = f.read()
  
    # just some splits som that the text aren't huge
    t1 = "\n".join([t for t in text.split("\n")[1:50] if t])
    t2 = "\n".join([t for t in text.split("\n")[50:100] if t])
    t3 = "\n".join([t for t in text.split("\n")[100:150] if t])

    # we will test it using a list but a single text will work as well
    texts = [t1, t2, t3]
    return texts