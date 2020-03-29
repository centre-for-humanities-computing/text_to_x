"""
"""
from inspect import signature

import pandas as pd

from tensorflow.keras.preprocessing.text import text_to_word_sequence


def keras_tokenize_wrapper(texts,
                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower=False,
                           split=' ',
                           **kwargs):
    """

    Examples
    >>> texts = get_test_data()
    >>> dfs = keras_tokenize_wrapper(texts)
    >>> isinstance(dfs[0], pd.DataFrame)
    True
    """

    # get valid kwargs - only pass on valid arguments
    ttws_kwargs = {k: kwargs[k] for k in kwargs if
                   k in signature(text_to_word_sequence).parameters}

    dfs = [pd.DataFrame(text_to_word_sequence(text,
                                              filters=filters,
                                              lower=lower,
                                              split=split,
                                              **ttws_kwargs),
                        columns=['token']) for text in texts]
    return dfs


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
