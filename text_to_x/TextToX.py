"""
"""

from text_to_x.utils import detect_lang_polyglot


class TextToX():
    def __init__(self, lang, kwargs, detect_lang_fun="polyglot"):
        """
        Super class for the text_to transformer classes.
        Provides language detection with more.

        If language should not be detected, set 'lang = -1'.
        """
        self.lang = lang
        self._kwargs = kwargs
        self._detect_lang_fun = detect_lang_fun
        self.__prepare_language_detector()

    def get_lang(self):
        return self.lang


    # Private methods
    def __prepare_language_detector(self):
        if self.lang == -1:
            # Child class does not use language
            return None
        self.__detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        if isinstance(self._detect_lang_fun, str):
            if self._detect_lang_fun not in self.__detect_lang_fun_dict:
                raise ValueError(f"{self._detect_lang_fun} is not a valid\
                                  string for detect_lang_fun")
            self._detect_lang_fun = self.\
                __detect_lang_fun_dict[self._detect_lang_fun]

        elif not callable(self._detect_lang_fun):
            raise TypeError(f"detect_lang_fun should be a string or callable\
                not a {type(self._detect_lang_fun)}")

    # Protected methods (Accessible by subclasses)
    def _get(self, x, msg, err=RuntimeError):
        """
        Create default getter method.
        If x is None, raise error. Otherwise, return x.
        """
        if x is None:
            raise err(msg)
        return x

    def _detect_language(self, texts, simplify=True):
        """
        Detect the language of each text.
        simplify (bool): When all texts have the same language, whether to set
        to scalar string with that code.
        """
        if self.lang is None:
            self.lang = [self._detect_lang_fun(text, **self._kwargs)
                         for text in texts]
        if simplify:
            self._simplify_lang()

    def _simplify_lang(self):
        """
        If self.lang is a list with exactly ONE unique language code,
        set to be that code.
        """
        if isinstance(self.lang, list) and len(set(self.lang)) == 1:
            self.lang = self.lang[0]
