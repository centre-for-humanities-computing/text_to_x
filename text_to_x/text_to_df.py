"""
"""
import types

import numpy as np

from text_to_x.utils import detect_lang_polyglot
from text_to_x.methods.stanfordnlp_to_df import stanfordnlp_to_df

class TextToDf():
    def __init__(self, lang = None, method = "stanfordnlp", args = {"processor":"tokenize,mwt,lemma,pos,depparse"}, detect_lang_fun = "polyglot", **kwargs):
        """
        lang (str): language code, if None language is detected using detect_lang_fun (which defaults to polyglot).
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return 
        method (str|fun): method used for normalization
        args (dict): arguments to be passed to the method
        """
        self.__detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        self.__method_dict = {"stanfordnlp": stanfordnlp_to_df}

        if isinstance(method, str):
            self.method = self.__method_dict[method]
        elif not callable(method):
            raise TypeError(f"method should be a str or callable not a type: {type(method)}")

        if isinstance(detect_lang_fun, str):
            self.detect_lang_fun = self.__detect_lang_fun_dict[detect_lang_fun]
        elif not callable(detect_lang_fun):
            raise TypeError(f"detect_lang_fun should be a string or callable not a {type(detect_lang_fun)}")

        self.lang = lang
        self.args = args
        self.kwargs = kwargs

    def texts_to_dfs(self, texts):
        """
        texts (str|list): Should be a string, a list or other iterable object
        """
        if isinstance(texts, str):
            texts = [texts]

        if self.lang is None:
            self.lang = [self.detect_lang_fun(text, **self.kwargs) for text in texts]
        
        self.texts = texts
        self.dfs = self.method(texts, self.lang, **self.args)
        return self.dfs

def texts_to_dfs(texts, lang = None, method = "stanfordnlp", args = {"processor":"tokenize,mwt,lemma,pos,depparse"}):
    """
    texts (str|list): Should be a string, a list or other iterable object
    method (str|fun): method used for normalization, currently implemted
    args (dict): arguments to be passed to the method
    """
    ttd = TextToDf(lang = lang, method = method, args = args)
    return ttd.texts_to_dfs(texts = texts)

    
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
