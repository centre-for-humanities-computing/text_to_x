"""

"""
import warnings

import pandas as pd

from text_to_x.text_to_tokens import TextToTokens
from text_to_x.text_to_sentiment import TextToSentiment
from text_to_x.text_to_vocab import TextToVocab
from text_to_x.text_to import TextTo

class Texts(TextTo):
    def __init__(self, texts, 
                 lang = None,  
                 detect_lang_fun = "polyglot", 
                 **kwargs):
        """
        texts (list): texts to process.
        lang (str): language code(s), if None language is detected using detect_lang_fun (which defaults to polyglot). Can be list of codes.
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return   

        Examples
        >>> with open("test_data/fyrtårnet.txt", "r") as f: # load data
        ...     text = f.read()
        >>> t1 = '\\n'.join([t for t in text.split('\\n')[1:50] if t]) # take subset of data
        >>> t2 = '\\n'.join([t for t in text.split('\\n')[50:100] if t])
        >>> texts = [t1, t2]
        >>> tt = Texts(texts, lang = "da") # make text class
        >>> tt.preprocess(silent = True)
        ...
        >>> dfs = tt.get_preprocessed_texts()
        >>> isinstance(dfs, list)
        True
        >>> isinstance(dfs[0], pd.DataFrame)
        True
        >>> tt.score_sentiment()
        >>> df = tt.get_sentiments()
        >>> isinstance(df, pd.DataFrame)
        True
        """
        super().__init__(lang = lang, kwargs = kwargs, 
                         detect_lang_fun = detect_lang_fun)

        self.raw_texts = texts
        self.__preprocess_method = None
        self.preprocessors = None
        self.__sentiment_scores = None
        self.__preprocessed_texts = None
        self.is_preprocessed = False
        self.__token_counters = None
        self.__vocabs = None
        
        # Detect language if not specified
        self._detect_language(self.raw_texts)
        
    def __check_preprocessed(self, method_name, required_processes = None):
        assert required_processes is None or isinstance(required_processes, (str, list)), \
            "required_processes must be a list of strings or None."
        if not self.is_preprocessed:
            raise RuntimeError(f"{method_name} requires the preprocessing() method to be run first.")
        if required_processes is not None and not set(required_processes).issubset(
            set(self.applied_preprocessors)):
            raise RuntimeError(f"{method_name} requires these preprocessing steps: {required_processes}")

    def preprocess(self, 
                   preprocess_method = "stanza",
                   lemmatize = True,
                   stem = False,
                   pos = True,
                   mwt = False,
                   depparse = False,
                   casing = False,
                   silent = False):
        """
        preprocess_method (str|fun): method used for normalization
        preprocessors (list): names of processes to apply in the preprocessing stage

        Note: Overwrites previous preprocessing!
        """

        if self.is_preprocessed:
            warnings.warn("Overwriting previous preprocessing.")

        self.__preprocess_method = preprocess_method
        self.preprocessors = {
            "tokenize" : True,
            "lemma" : lemmatize,
            "stem" : stem,
            "mwt" : mwt, 
            "pos" : pos, 
            "depparse" : depparse,
            "casing" : casing
        }
        self.applied_preprocessors = [procss for procss,flag in self.preprocessors.items() if flag]
        self.__preprocessed_ttt = TextToTokens(
            lang = self.lang, 
            method = self.__preprocess_method, 
            lemmatize = lemmatize,
            stem = stem,
            pos = pos,
            mwt = mwt,
            depparse = depparse,
            casing = casing)
        self.__preprocessed_texts = self.__preprocessed_ttt.texts_to_tokens(
            texts = self.raw_texts, silent = silent)
        self.is_preprocessed = True

    def get_preprocessed_texts(self):
        return self._get(self.__preprocessed_texts, 
                         "The preprocess() method has not been called yet.")

    def extract_vocabulary(self, type_token = "token", lower = False):
        """
        Creates a token counter (term frequency) and a vocabulary for each text.
        Get result with get_vocabularies() or get_token_counters().

        type_token (str): Either 'token', 'lemma', or 'stem'.
        lower (bool): Whether to lowercase the tokens first.
        """
        if type_token == "token":
            required_process = "tokenize"
        else:
            required_process = type_token
        self.__check_preprocessed("extract_vocabulary()", [required_process])
        ttv = TextToVocab(type_token = type_token)
        self.__token_counters, self.__vocabs = ttv.texts_to_vocabs(
            self.__preprocessed_texts, lower = lower)

    def get_vocabularies(self):
        return self._get(self.__vocabs, 
                         "The extract_vocabulary() method has not been called yet.")

    def get_token_counters(self):
        return self._get(self.__token_counters, 
                         "The extract_vocabulary() method has not been called yet.")

    def score_sentiment(self, method = "dictionary", type_token = None):
        """
        method ("dictionary"|"bert"|fun): method used for sentiment analysis
        type_token (None|'lemma'|'token'): The type of token used. If None is chosen to be token automatically depending on method.
          'lemma' for dictionary otherwise 'token'.

        Requires these preprocessing steps: 
          "tokenize", "lemma"

        Use get_sentiments() method to extract scores.
        """
        self.__check_preprocessed("score_sentiment()", ["tokenize","lemma"])
        tts = TextToSentiment(lang=self.lang, method=method, type_token=type_token)
        self.__sentiment_scores = tts.texts_to_sentiment(self.__preprocessed_ttt)

    def get_sentiments(self):
        return self._get(self.__sentiment_scores, 
                         "The score_sentiment() method has not been called yet.")

    def model_topics(self):
        # self.__check_preprocessed("model_topics()", ["tokenize","lemma"])
        pass

    def get_topics(self):
        pass



# testing code
if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)

if __name__ == "__main__":
    import os
    os.getcwd()
    os.chdir("..")
    # make some data
    from text_to_x.utils import get_test_data
    texts = get_test_data()

    # Init Texts object
    tt = Texts(texts, lang = "da")
    tt.preprocess(silent = True)
    dfs = tt.get_preprocessed_texts()
    # Extract vocabulary and token counter (term frequency)
    tt.extract_vocabulary()
    tf = tt.get_token_counters()
    vcb = tt.get_vocabularies()
    # Score sentiment
    tt.score_sentiment()
    df = tt.get_sentiments()
    # Topic modeling
    # tt.model_topics()

    