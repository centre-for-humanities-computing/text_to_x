"""
"""
import types

import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import PorterStemmer

from text_to_x.utils import detect_lang_polyglot, silence
from text_to_x.methods.stanza_to_df import stanza_to_df



class TextToTokens():
    def __init__(self, lang = None, method = "stanza", args = ["tokenize", "mwt", "lemma", "pos", "depparse", "ner", "stem"], detect_lang_fun = "polyglot", **kwargs):
        """
        lang (str): language code, if None language is detected using detect_lang_fun (which defaults to polyglot).
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return 
        method (str|fun): method used for normalization
        args (list): can include "tokenize", "mwt", "lemma", "pos", "depparse", "stem"
        """
        self.__detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        self.__method_dict = {"stanza": stanza_to_df}
        self.lang = lang
        self.args = args
        self.dfs = None
        self.kwargs = kwargs

        if 'tokenize' not in args and "stem" in args:
            args.append('tokenize')
        
        self.preprocessor_args = {"processors": ",".join(a for a in self.args if a in {"tokenize", "mwt", "lemma", "pos", "depparse", "ner"})}

        if isinstance(method, str):
            self.method = self.__method_dict[method]
        elif not callable(method):
            raise TypeError(f"method should be a str or callable not a type: {type(method)}")

        if isinstance(detect_lang_fun, str):
            self.detect_lang_fun = self.__detect_lang_fun_dict[detect_lang_fun]
        elif not callable(detect_lang_fun):
            raise TypeError(f"detect_lang_fun should be a string or callable not a {type(detect_lang_fun)}")

    def texts_to_tokens(self, texts, silent = True):
        """
        texts (str|list): Should be a string, a list or other iterable object
        """
        if isinstance(texts, str):
            texts = [texts]

        if silent:
            sav = self.method
            self.method = silence(self.method)

        if self.lang is None:
            self.lang = [self.detect_lang_fun(text, **self.kwargs) for text in texts]
        
        self.texts = texts
        self.dfs = self.method(texts, self.lang, **self.preprocessor_args)
        if 'stem' in self.args:
            self.__stem(**self.kwargs)
        if silent:
            self.method = sav
        return self.dfs

    def __get_stemmer(self, stemmer, lang):
        """
        method (str): method for stemming, can be either snowball or porter
        """
        lang_dict = {"da":"danish", "en": "english"}
        lang = lang_dict.get(lang, lang)
        if stemmer == "porter":
            ps = PorterStemmer()
            self.stemmer = ps.stem
        elif stemmer == "snowball":
            ss = SnowballStemmer(lang)
            self.stemmer = ss.stem
        elif not callable(self.stemmer):
            raise TypeError(f"stemmer should be a 'porter' or 'snowball' or callable not a type: {type(self.stemmer)}")
      

    def __stem(self, stemmer = "snowball", **kwargs):
        if isinstance(self.lang, str):
            self.__get_stemmer(stemmer, self.lang)
            for df in self.dfs:
                df['stem'] = [self.stemmer(token) for token in df['token']]
        else:
            for i, l in enumerate(self.lang):
                if i == 0:
                    lang = l
                    self.__get_stemmer(stemmer, lang)
                elif l != lang:
                    self.__get_stemmer(stemmer, lang)
                self.dfs[i]['stem'] = [self.stemmer(token) for token in self.dfs[i]['token']]



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

    ttt = TextToTokens(lang = "da")
    dfs = ttt.texts_to_tokens(texts)