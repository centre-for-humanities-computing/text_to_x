"""
"""
import types

import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import PorterStemmer

from text_to_x.utils import silence
from text_to_x.methods.stanza_to_df import stanza_to_df
from text_to_x.text_to import TextTo


class TextToTokens(TextTo):
    def __init__(self, lang = None, 
                 method = "stanza",
                 lemmatize = True,
                 stem = False,
                 pos = False,
                 mwt = False,
                 depparse = False,
                 casing = False,
                 silent = False,
                 detect_lang_fun = "polyglot", 
                 **kwargs):
        """
        lang (str): language code(s), if None language is detected using detect_lang_fun (which defaults to polyglot). Can be a list of codes.
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return 
        method (str|fun): method used for normalization
        args (list): can include "tokenize", "mwt", "lemma", "pos", "depparse", "stem"
        """
        super().__init__(lang = lang, kwargs = kwargs, 
                         detect_lang_fun = detect_lang_fun)
        
        self.preprocessors = {
            "tokenize" : True,
            "lemma" : lemmatize,
            "stem" : stem,
            "mwt" : mwt, 
            "pos" : pos, 
            "depparse" : depparse,
            "casing" : casing
        }
        self.__preprocessor_args = {"processor": ",".join(
            [procss for procss, flag in self.preprocessors.items() if \
                flag and procss not in ["stem","casing"]])}
        
        self.dfs = None

        self.__method_dict = {"stanza": stanza_to_df}
        if isinstance(method, str):
            self.method = self.__method_dict[method]
        elif not callable(method):
            raise TypeError(f"method should be a str or callable not a type: {type(method)}")

    def texts_to_tokens(self, texts, silent = True):
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
        if self.preprocessors['stem']:
            self.__stem(**self._kwargs)
        if self.preprocessors['casing']:
            self.__extract_casing()
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

    def __extract_casing(self):
        """
        Whether token is title cased, upper cased, lower cased, or mixed cased.
        Note: Title cased is also mixed cased.
        """
        def casings_single_df(df):
            df['title_cased'] = [int(token.istitle()) for token in df['token']]
            df['upper_cased'] = [int(token.isupper()) for token in df['token']]
            df['lower_cased'] = [int(token.islower()) for token in df['token']]
            df['mixed_cased'] = df.apply(
                lambda r: int((r['upper_cased']+r['lower_cased']) == 0), 
                axis = 1)
            return df
        self.dfs = [casings_single_df(df) for df in self.dfs]
            

        
if __name__ == "__main__":
    # testing code

    # make some data
    from text_to_x.utils import get_test_data
    texts = get_test_data()

    ttd = TextToTokens(lang = "da")
    dfs = ttd.texts_to_tokens(texts)

