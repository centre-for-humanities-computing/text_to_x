"""
"""
import types

import numpy as np

from utils import detect_lang_polyglot
from methods.stanfordnlp_to_df import stanfordnlp_to_df


class TextToDf():
    def __init__(self, texts, detect_lang_fun = "polyglot", lang = None, **kwargs):
        """
        texts (str|list): Should be a string, a list or other iterable object
        lang (str): language code, if None language is detected using polyglot.
        """

        self.__detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        self.__method_dict = {"stanfordnlp": stanfordnlp_to_df}

        if isinstance(texts, str):
            texts = [texts]



        if isinstance(detect_lang_fun, str):
            detect_lang_fun = self.__detect_lang_fun_dict[detect_lang_fun]
        elif not callable(detect_lang_fun):
            raise ValueError(f"detect_lang_fun should be a string or callable not a {type(detect_lang_fun)}")


        if lang is None:
            self.lang = [detect_lang_polyglot(text)[0] for text in texts]
        else:
            self.lang = lang

        self.texts = texts

    def text_to_df(self, method = "stanfordnlp", args = {"processor":"tokenize,mwt,lemma,pos,depparse"}):
        """
        method (str|fun): method used for normalization
        args: arguments to be passed to the method
        """

        if isinstance(method, str):
            method = self.__method_dict[method]
        elif not callable(method):
            ValueError(f"method should be a str or callable not a type: {type(method)}")
        
        self.dfs = method(self.texts, self.lang, **args)
        return self.dfs

def texts_to_dfs(texts, lang = None, method = "stanfordnlp", args = {"processor":"tokenize,mwt,lemma,pos,depparse"}, **kwargs):
    """


    method (str|fun): method used for normalization, currently implemted
    args: arguments to be passed to the method
    """
    ttd = TextToDf(texts = text, lang = lang, **kwargs)
    return ttd.text_to_df(method = method, **kwargs)

    
if __name__ == "__main__":
    # testing code

    # make some data
    with open("test_data/fyrtårnet.txt", "r") as f:
        text = f.read()
        # just some splits som that the text aren't huge
    t1 = "\n".join([t for t in text.split("\n")[1:50] if t])
    t2 = "\n".join([t for t in text.split("\n")[50:100] if t])
    t3 = "\n".join([t for t in text.split("\n")[100:150] if t])

    # we will test it using a list but a single text will work as well
    texts = [t1, t2, t3]

    dfs = texts_to_dfs(texts)
    dfs[0]
