"""
"""
import os
import sys

import numpy as np


def detect_lang_polyglot(text,
                         simplify=True,
                         print_error=False,
                         raise_error=True,
                         keep_unreliable=False, **kwargs):
    """
    For detecting language using polyglot, but with exception handling

    Examples:
    >>> detect_lang("This is a test text")
    ('en', 95.0)
    >>> detect_lang(text = "Dette er åbenbart en norsk tekst",\
      keep_unreliable = True)
    ('no', 97.0)
    >>> detect_lang(text = "Dette er åbenbart en norsk tekst. \
      This is also an english text.", keep_unreliable = True)
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
    """
    func (fun): function which you desire silences

    Examples
    >>> def addn(x,n):  # function with annoying print
    ...    print(f"adding {n} to {x}")
    ...    return x + n
    >>> add_silences = silence(addn)
    >>> add_silences(3, 1)
    4
    """
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sav = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sav
        return value
    return func_wrapper


def add_def_args(func, def_args):
    """
    func (fun): function which to add defaults arguments to
    def_args (dict): default argument given as a dictionary

    Examples
    >>> def addn(x,n):
    ...    return x + n
    >>> add3 = add_def_args(addn, {'n':3})
    >>> add3(2)
    5
    """
    def func_wrapper(*args, **kwargs):
        value = func(*args, **def_args, **kwargs)
        return value
    return func_wrapper


def add_module_to_path(module="text_to_x"):
    # path to file
    p = os.path.dirname(os.path.abspath(__file__))
    # the highest level in path which corresponds to module
    path_n = max([i for i, d in enumerate(p.split("/")) if d == module])
    # recreate path
    path = "/".join(p.split("/")[:path_n])
    # add to module to path
    sys.path.append(path)


 p = os.path.dirname(os.path.abspath(__file__))
path_n = max([i for i, d in enumerate(p.split("/")) if d == "text_to_x"])
path = "/".join(p.split("/")[:path_n+1])
read_path = path + "/test_data/fyrtårnet.txt"
with open(path + "/test_data/fyrtårnet.txt", "r") as f:
    ft = f.read()
with open(path + "/test_data/origin_of_species.txt", "r") as f:
    oos = f.read()
def get_test_data(data="fyrtårnet", short_splits=True):
    """
    data ('fyrtårnet'|'origin_of_species')
    """

    if data == 'fyrtårnet':
        text = ft
    elif data == 'origin_of_species':
        text = oos
    else:
        ValueError(f"{data} is not a valid argument for data.")


    if short_splits is True:
        # just some splits som that the text aren't huge
        t1 = "\n".join([t for t in text.split("\n")[1:50] if t])
        t2 = "\n".join([t for t in text.split("\n")[50:100] if t])
        t3 = "\n".join([t for t in text.split("\n")[100:150] if t])

        # we will test it using a list but a single text will work as well
        texts = [t1, t2, t3]
        return texts
    return text


# Example of typecheck
# import typecheck as tc
# @tc.typecheck
# def foo(x: str) -> str:
#     return x.split()

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
