
from text_to_x.utils import detect_lang_polyglot

class TextTo():
    def __init__(self, lang, kwargs, detect_lang_fun = "polyglot"):
        """
        Super class for the text_to transformer classes.
        Provides language detection with more.
        """
        self.lang = lang
        self._kwargs = kwargs
        self._detect_lang_fun = detect_lang_fun
        self.__prepare_language_detector()
        
    # Private methods

    def __prepare_language_detector(self):
        self.__detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        if isinstance(self._detect_lang_fun, str):
            if self._detect_lang_fun not in self.__detect_lang_fun_dict:
                raise ValueError(f"{self._detect_lang_fun} is not a valid string for detect_lang_fun")
            self._detect_lang_fun = self.__detect_lang_fun_dict[self._detect_lang_fun]
        elif not callable(self._detect_lang_fun):
            raise TypeError(f"detect_lang_fun should be a string or callable not a {type(self._detect_lang_fun)}")

    # Protected methods (Accessible by subclasses)

    def _get(self, x, msg, err = RuntimeError):
        """
        Create default getter method. 
        If x is None, raise error. Otherwise, return x.
        """
        if x is None:
            raise err(msg)
        return x

    def _detect_language(self, texts):
        """
        Detect the language of each text.
        """
        if self.lang is None:
            self.lang = [self._detect_lang_fun(text, **self._kwargs) for text in texts]
