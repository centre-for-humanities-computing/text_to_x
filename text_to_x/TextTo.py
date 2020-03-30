"""

"""
import warnings

import pandas as pd

from text_to_x.TextToTokens import TextToTokens
from text_to_x.TextToSentiment import TextToSentiment
from text_to_x.TextToVocab import TextToVocab
from text_to_x.TextToX import TextToX
from text_to_x.concordance import extract_concordance


class TextTo(TextToX):
    def __init__(self, texts,
                 lang=None,
                 detect_lang_fun="polyglot",
                 **kwargs):
        """
        texts (list): texts to process.
        lang (str): language code(s), if None language is detected using
        detect_lang_fun (which defaults to polyglot). Can be list of codes.
        detect_lang_fun (str|fun): fucntion to use for language detection.
        default is polyglot. But you can specify a user function, which return

        Examples
        >>> with open("test_data/fyrtårnet.txt", "r") as f: # load data
        ...     text = f.read()
        >>> t1 = '\\n'.join([t for t in text.split('\\n')[1:50] if t])
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
        super().__init__(lang=lang, kwargs=kwargs,
                         detect_lang_fun=detect_lang_fun)

        self.raw_texts = texts
        self.__preprocess_method = None
        self.preprocessors = None
        self.__sentiment_scores = None
        self.__preprocessed_texts = None
        self.is_preprocessed = False
        self.__token_counters = None
        self.__vocabs = None
        self.__concordances = None
        self.__descriptors = None

        # Detect language if not specified
        self._detect_language(self.raw_texts)

    def __check_preprocessed(self, method_name, required_processes=None):
        assert required_processes is None or\
            isinstance(required_processes, (str, list)), \
            "required_processes must be a list of strings or None."
        if not self.is_preprocessed:
            raise RuntimeError(f"{method_name} requires the preprocessing() \
                method to be run first.")
        if required_processes is not None and not set(required_processes).\
                issubset(set(self.applied_preprocessors)):
            raise RuntimeError(f"{method_name} requires these preprocessing \
                steps: {required_processes}")

    # Preprocessing

    def preprocess(self,
                   preprocess_method="stanza",
                   lemmatize=True,
                   stem=False,
                   pos=True,
                   mwt=False,
                   depparse=False,
                   casing=False,
                   silent=False):
        """
        preprocess_method (str|fun): method used for normalization
        preprocessors (list): names of processes to apply in the preprocessing
        stage

        Note: Overwrites previous preprocessing!
        # TODO: fix this function
        """

        if self.is_preprocessed:
            warnings.warn("Overwriting previous preprocessing.")

        self.__preprocess_method = preprocess_method
        self.preprocessors = {
            "tokenize": True,
            "lemma": lemmatize,
            "stem": stem,
            "mwt": mwt,
            "pos": pos,
            "depparse": depparse,
            "casing": casing
        }
        self.applied_preprocessors = [procss for procss, flag in
                                      self.preprocessors.items() if flag]
        self.__preprocessed_ttt = TextToTokens(
            lang=self.lang,
            method=self.__preprocess_method,
            lemmatize=lemmatize,
            stem=stem,
            pos=pos,
            mwt=mwt,
            depparse=depparse,
            casing=casing)
        self.__preprocessed_texts = self.__preprocessed_ttt.texts_to_tokens(
            texts=self.raw_texts, silent=silent)
        self.is_preprocessed = True

    def get_preprocessed_texts(self):
        return self._get(self.__preprocessed_texts,
                         "The preprocess() method has not been called yet.")

    # Vocabulary

    def extract_vocabulary(self, type_token="token", lower=False, top_n=None):
        """
        Creates a token counter (term frequency) and a vocabulary for each
        text.
        Get result with get_vocabularies() or get_token_counters().

        type_token (str): Either 'token', 'lemma', or 'stem'.
        lower (bool): Whether to lowercase the tokens first.
        top_n (int | None): Keep only the top n most frequent tokens.
        """
        if type_token == "token":
            required_process = "tokenize"
        else:
            required_process = type_token
        self.__check_preprocessed("extract_vocabulary()", [required_process])
        ttv = TextToVocab(type_token=type_token)
        self.__token_counters, self.__vocabs = ttv.texts_to_vocabs(
            self.__preprocessed_texts, lower=lower, top_n=top_n)

    def get_vocabularies(self):
        return self._get(self.__vocabs,
                         "The extract_vocabulary() method has not been called \
                             yet.")

    def get_token_counters(self):
        return self._get(self.__token_counters,
                         "The extract_vocabulary() method has not been called \
                             yet.")

    # Concordance

    def extract_concordance(self, tokens, type_token="token", lower=True):
        """
        For a set of tokens, extract the sentences they occur in.

        tokens (list of str): List of tokens to find concordances for.
        type_token (str): Either 'token', 'lemma', or 'stem'.
        lower (bool): Whether to match the tokens in lowercase.
          Returned sentences will have original case.
        """
        self.__concordances = extract_concordance(
            self.__preprocessed_texts, tokens=tokens,
            type_token=type_token, lower=lower)

    def get_concordances(self):
        return self._get(self.__concordances,
                         "The extract_concordance() method has not been called \
                             yet.")

    # Text Descriptors

    def calculate_descriptive_stats(self):
        from textdescriptives import all_metrics
        if isinstance(self.lang, str):
            self.__descriptors = all_metrics(self.raw_texts, lang=self.lang)
        elif isinstance(self.lang, list):
            # TODO Change if Lasse enables a list of languages
            # TODO Otherwise, call for each collection of languages and
            # restore order afterwards
            self.__descriptors = pd.concat(
                [all_metrics(t, lang=l)
                 for t, l in zip(self.raw_texts, self.lang)])

    def get_descriptive_stats(self):
        return self._get(self.__descriptors,
                         "The calculate_descriptive_stats() method has not \
                             been called yet.")

    # Sentiment

    def score_sentiment(self, method="dictionary", type_token=None):
        """
        method ("dictionary"|"bert"|fun): method used for sentiment analysis
        type_token (None|'lemma'|'token'): The type of token used. If None is
        chosen to be token automatically depending on method.
          'lemma' for dictionary otherwise 'token'.

        Requires these preprocessing steps:
          "tokenize", "lemma"

        Use get_sentiments() method to extract scores.
        """
        self.__check_preprocessed("score_sentiment()", ["tokenize", "lemma"])
        tts = TextToSentiment(lang=self.lang, method=method,
                              type_token=type_token)
        self.__sentiment_scores = tts.\
            texts_to_sentiment(self.__preprocessed_ttt)

    def get_sentiments(self):
        return self._get(self.__sentiment_scores,
                         "The score_sentiment() method has not been called\
                             yet.")

    # Topic modeling

    def model_topics(self):
        # self.__check_preprocessed("model_topics()", ["tokenize","lemma"])
        pass

    def get_topics(self):
        pass


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
    tt = TextTo(texts, lang="da")
    tt.preprocess(silent=True)
    dfs = tt.get_preprocessed_texts()
    # Extract vocabulary and token counter (term frequency)
    tt.extract_vocabulary(top_n=20, lower=True)
    tf = tt.get_token_counters()
    vcb = tt.get_vocabularies()
    # Extract concordance (sentences the token occur in)
    tt.extract_concordance(tokens=["kammer", "skilling", "soldaterne"],
                           type_token="token", lower=True)
    cc = tt.get_concordances()
    # Calculate descriptive stats
    tt.calculate_descriptive_stats()
    ds = tt.get_descriptive_stats()
    # Score sentiment
    tt.score_sentiment()
    df = tt.get_sentiments()
    # Topic modeling
    # tt.model_topics()
