"""

"""

import pandas as pd
from collections import Counter

from text_to_x.text_to_tokens import TextToTokens
from text_to_x.text_to import TextTo


class TextToVocab(TextTo):
    def __init__(self, type_token = "token", **kwargs):
        """
        Extract vocabulary and term counter for each text.

        type_token (str): Either 'token', 'lemma', or 'stem'.
        """
        super().__init__(lang = -1, kwargs = kwargs)
        # We don't use the language in this class

        assert type_token in ["token", "lemma", "stem"], \
            "type_token must be one of 'token', 'lemma', and 'stem'."

        self.type_token = type_token
        self.__counters = None
        self.__vocabs = None
        
    def texts_to_vocabs(self, preprocessed_texts, lower = False, top_n = None): # TODO change first arg name?
        """
        preprocessed_texts (list of data frames | TextToTokens): Data frames with tokens.
        lower (bool): Whether to lowercase the tokens first.
        top_n (int | None): Keep only the top n most frequent tokens.
        
        Returns 1) list of collections.Counter objects and 2) lists of unique tokens
        """
        if isinstance(preprocessed_texts, TextToTokens):
            preprocessed_texts = preprocessed_texts.get_token_dfs()
        elif isinstance(preprocessed_texts, list) and not isinstance(preprocessed_texts[0], pd.DataFrame):
            raise TypeError("When preprocessed_texts is a list, it must contain data frames.")
        assert self.type_token in preprocessed_texts[0].columns, \
            "type_token wasn't a column in the first preprocessed text data frame."

        def lower_tokens(tokens):
            if lower:
                return [tok.lower() for tok in tokens]
            return tokens

        self.__counters = [Counter(lower_tokens(df[self.type_token])) for df in preprocessed_texts]
        if top_n is not None:
            self.__counters = [Counter(dict(c.most_common(top_n))) for c in self.__counters]
        self.__vocabs = [sorted(list(c.keys())) for c in self.__counters]
        return self.__counters, self.__vocabs

    def get_counters(self):
        return self._get(self.__counters,
                         "The texts_to_vocabs() method has not been called yet.")

    def get_vocabularies(self):
        return self._get(self.__vocabs,
                         "The texts_to_vocabs() method has not been called yet.")

        

