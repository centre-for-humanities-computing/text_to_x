"""
"""
import types

import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import PorterStemmer

from text_to_x.utils import silence
from text_to_x.methods.stanfordnlp_to_df import stanfordnlp_to_df
from text_to_x.text_to import TextTo


class TextToDf(TextTo):
    def __init__(self, lang = None, method = "stanfordnlp", 
                 args = ["tokenize", "mwt", "lemma", "pos", "depparse", "stem"], 
                 detect_lang_fun = "polyglot", **kwargs):
        """
        lang (str): language code(s), if None language is detected using detect_lang_fun (which defaults to polyglot). Can be a list of codes.
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return 
        method (str|fun): method used for normalization
        args (list): can include "tokenize", "mwt", "lemma", "pos", "depparse", "stem"
        """
        super().__init__(lang = lang, kwargs = kwargs, 
                         detect_lang_fun = detect_lang_fun)
        
        self.__method_dict = {"stanfordnlp": stanfordnlp_to_df}
        self.args = args
        self.dfs = None

        if 'tokenize' not in args:
            args.append('tokenize')
        self.__preprocessor_args = {"processor": ",".join(self.args)}

        if isinstance(method, str):
            self.method = self.__method_dict[method]
        elif not callable(method):
            raise TypeError(f"method should be a str or callable not a type: {type(method)}")

    def texts_to_dfs(self, texts, silent = True):
        """
        texts (str|list): Should be a string, a list or other iterable object
        """
        if isinstance(texts, str):
            texts = [texts]

        # Detect language if not specified
        self._detect_language(texts)

        if silent:
            sav = self.method
            self.method = silence(self.method)

        self.texts = texts
        self.dfs = self.method(texts, self.lang, **self.__preprocessor_args)
        if 'stem' in self.args:
            self.__stem(**self._kwargs)
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

    ttd = TextToDf(lang = "da")
    dfs = ttd.texts_to_dfs(texts)

