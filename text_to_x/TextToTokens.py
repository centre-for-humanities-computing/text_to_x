"""
"""
import re
from warnings import warn

import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.snowball import PorterStemmer

from text_to_x.TextToX import TextToX
from text_to_x.utils import silence, add_def_args
from text_to_x.methods.stanza_to_df import stanza_to_df
from text_to_x.methods.tokenize import keras_tokenize_wrapper
from text_to_x.methods.flair_ner import flair_tagger_ttt


class TextToTokens(TextToX):
    def __init__(self, lang=None,
                 tokenize="stanza",
                 lemmatize="stanza",
                 stemming=None,
                 pos="stanza",
                 mwt=None,
                 depparse=None,
                 ner=None,
                 casing=False,
                 detect_lang_fun="polyglot",
                 **kwargs):
        """
        lang (str): language code(s), if None language is detected using
        detect_lang_fun (defaults to polyglot). Can be a list of codes. If it
        is a list of codes, it should be the same length as the list of
        texts.
        tokenize (str|fun): the function to use for tokenization. Should return
        a dataframe with column called 'token'. Current implemented methods:
        'stanza', 'keras'
        keras is the fasted method, but used regular expressions.
        lemmatize (str|fun): the function to use for lemmatization. Should
        add to column 'lemma' to a dataframe using a token column. Current
        implemented methods:
        'stanza'
        stemming (str|fun): the function to use for stemming. Should
        add to column 'stem' to a dataframe using a token column. Current
        implemented methods:
        'snowball', 'porter'
        pos (str|fun): the function to use for part-of-speech tagging
        (pos-tagging). Should add to column 'pos', 'xpos' or 'upos' depending
        on type of pos-tag to a dataframe. Current implemented methods:
        'stanza'
        mwt (str|fun): multi-word-token expansion (mwt). Should take a
        dataframe with a column of tokens and return a multi word token
        expanded dataframe. Current implemented methods:
        'stanza'
        depparse (str|fun): the function to use for dependency parsing.
        Should add to column 'dependecy relation' to a dataframe. Current
        implemented methods:
        'stanza'
        ner (str|fun): the function to use for named entity recogniton (ner).
        Should add to column 'ner' to a dataframe. Current implemented methods:
        'stanza', 'flair'
        casing (bool): should casing be extracted? Default to False
        detect_lang_fun (str|fun): fucntion to use for language detection.
        default is polyglot. But you can specify a user function as well.
        kwargs arguments to be passed to any of the functions

        TODO:
        add doctest:
        stanza, both stemmers, keras tokenizer, casing
        """
        if not tokenize:
            warn("Tokenize evaluates to False, setting tokenize to 'keras' to \
                 allow for other processes to run")
            tokenize = "keras"

        super().__init__(lang=lang, kwargs=kwargs,
                         detect_lang_fun=detect_lang_fun)

        # define dict of valid preprocessors beside stanza
        preprocss_func = {
            "tokenize": {'keras': keras_tokenize_wrapper},
            "lemma": {},
            "stemming": {'snowball':
                         add_def_args(self.__stem, {'stemmer': 'snowball'}),
                         'porter':
                         add_def_args(self.__stem, {'stemmer': 'porter'})},
            "mwt": {},
            "pos": {},
            "depparse": {},
            "ner": {'flair':
                    add_def_args(flair_tagger_ttt, {'tagging': 'ner'})},
            "casing": {True: self.__extract_casing},
        }

        self.preprocessors = {
            "tokenize": tokenize,
            "lemma": lemmatize,
            "stemming": stemming,
            "mwt": mwt,
            "pos": pos,
            "depparse": depparse,
            "ner": ner,
            "casing": casing
        }

        if tokenize == "stanza":
            s_valid_procss = {"tokenize", "lemma", "mwt", "pos", "depparse",
                              "ner"}
            s_procss = [key for key in self.preprocessors if
                        self.preprocessors[key] == "stanza" and
                        (key in s_valid_procss)]
            # remove items from dict
            for p in s_procss:
                self.preprocessors[p] = None

            # add default args to processor
            def_args = {"processors": ",".join(s_procss)}
            self.preprocessors['tokenize'] = add_def_args(stanza_to_df,
                                                          def_args)

        self.__dfs = None
        self.texts = None
        self.procss_order = ["tokenize", "lemma", "stemming", "mwt", "pos",
                             "depparse", "ner", "casing"]

        # check input
        for key in self.procss_order:
            procss = self.preprocessors.get(key, None)
            if callable(procss) or (not procss):
                continue
            elif isinstance(procss, str):
                if procss in preprocss_func[key]:
                    self.preprocessors[key] = preprocss_func[key][procss]
                elif procss == "stanza":
                    ValueError(f"To use the stanza for {key} you will have to use stanza for\
                                 Tokenization as well.")
                else:
                    ValueError(f"'{procss}' is not a valid string for {key}.\
                                 Valid string arguments are: \
                                 preprocss_func[key].keys")
            else:
                raise ValueError(f"{key} should be None, a str or a function,\
                                   not an instance of type {type(procss)}")

    def texts_to_tokens(self, texts, n_cores=None, silent=True):
        """
        texts (str|list): Should be a string, a list or other iterable object
        silent (bool): should the function be silent defaults to False
        """
        if isinstance(texts, str):
            texts = [texts]

        # Detect language if not specified
        self._detect_language(texts)
        self.texts = texts

        for key in self.procss_order:
            procss = self.preprocessors.pop(key)
            if (procss is None) or (procss is False):
                continue
            if silent:
                sv = procss
                procss = silence(procss)
            if key == 'tokenize':
                self.__dfs = procss(texts=texts, langs=self.lang)
            else:
                self.__dfs = procss(texts=texts,
                                    dfs=self.__dfs,
                                    langs=self.lang)
            if silent:
                procss = sv

    def get_token_dfs(self):
        return self._get(self.__dfs, "The texts_to_tokens() method has not\
                                      been called yet.")

    def __remove_stopwords(self):
        # TODO Implement and add flag to texts_to_tokens()?
        # NOTE: Not sure where this functionality is best to have?
        # KCE: best suited as a function?
        pass

    def __get_stemmer(self, stemmer, lang):
        """
        method (str): method for stemming, can be either snowball or porter
        """
        lang_dict = {"da": "danish", "en": "english"}
        if lang in lang_dict:
            lang = lang_dict[lang]
        else:
            raise ValueError(f"language {lang} not in language dict for\
                stemmer")
        if stemmer == "porter":
            ps = PorterStemmer()
            self.stemmer = ps.stem
        elif stemmer == "snowball":
            ss = SnowballStemmer(lang)
            self.stemmer = ss.stem
        elif not callable(self.stemmer):
            raise TypeError(f"stemmer should be a 'porter' or 'snowball' or\
                            callable not a type: {type(self.stemmer)}")

    def __stem(self, stemmer, dfs, lang, **kwargs):
        if isinstance(lang, str):
            self.__get_stemmer(stemmer, lang)
            for df in dfs:
                df['stem'] = [self.stemmer(token) for token in df['token']]
        else:
            for i, la in enumerate(lang):
                if i == 0:
                    lang_ = la
                    self.__get_stemmer(stemmer, lang_)
                elif la != lang_:
                    self.__get_stemmer(stemmer, lang_)
                dfs[i]['stem'] = [self.stemmer(token) for token in
                                  dfs[i]['token']]
        return dfs

    @staticmethod
    def __extract_casing(dfs):
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
                axis=1)
            return df
        dfs = [casings_single_df(df) for df in dfs]
        return dfs

    def twitter_format(self):
        """
        call this function to deal with emoticons, hashtags (#) and at (@)
        it will add to 'ner' TWITTER_USER and HASHTAG
        and will change POS tag for emojis to SYM_EMOJI
        """

        # regex replies (@something)
        # twitter username can be letters or numbers
        at_pattern = r'\B@\w*[a-zA-Z0-9]+\w*'

        # regex hashtags (#something)
        # twitter hashtag can't start with a number
        hash_pattern = r'\B#\w*[a-zA-Z]+\w*'

        # regex emojis
        # unicode characters in certain range
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                "]+", flags=re.UNICODE)

        # regex url
        # this is a monster, but works wonders
        url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')

        for i, df in enumerate(self.__dfs):
            hashtag_idx = np.array(df['token'] == "#")

            idx_offset = np.insert(hashtag_idx, 0, False)[:-1]
            # add hashtag to token
            df.loc[idx_offset, ['token']] = '@' + df.loc[idx_offset, ['token']]
            df.loc[idx_offset, ['lemma']] = '@' + df.loc[idx_offset, ['lemma']]
            # remove hashtag columns
            df = df.loc[np.invert(hashtag_idx)]
            hashtag_idx = np.array(df['token'].str.match(hash_pattern))
            df.loc[hashtag_idx, ['ner']] = "HASHTAG"

            at_idx = np.array(df['token'] == "@")
            idx_offset = np.insert(at_idx, 0, False)[:-1]
            df.loc[idx_offset, ['token']] = '@' + df.loc[idx_offset, ['token']]
            df.loc[idx_offset, ['lemma']] = '@' + df.loc[idx_offset, ['lemma']]
            df = df.loc[np.invert(at_idx)]
            at_idx = np.array(df['token'].str.match(at_pattern))
            df['ner'][at_idx] = "TWITTER_USER"

            emoji_idx = np.array(df['token'].str.match(emoji_pattern))
            df.loc[emoji_idx, ['upos']] = "SYM_EMOJI"

            url_idx = np.array(df['token'].str.match(url_pattern))
            df.loc[url_idx, ['ner']] = "URL"
            self.__dfs[i] = df


if __name__ == "__main__":
    from text_to_x.utils import get_test_data

    # make some data
    texts = get_test_data()

    ttt = TextToTokens(lang="da")
    ttt.texts_to_tokens(texts)
    dfs = ttt.get_token_dfs()
    dfs[0].head()
    dfs[0]

    ttt = TextToTokens(lang="da",
                       tokenize="keras",
                       lemmatize=None,
                       pos=None,
                       ner="flair")
    ttt.texts_to_tokens(texts)
    dfs = ttt.get_token_dfs()
    dfs[0].head()
