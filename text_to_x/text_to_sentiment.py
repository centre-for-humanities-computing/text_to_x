"""
"""
import pandas as pd

from text_to_x.vaderSentiment.vaderSentiment_en import SentimentIntensityAnalyzer as Sentiment_en
from text_to_x.vaderSentiment.vaderSentiment_da import SentimentIntensityAnalyzer as Sentiment_da
from text_to_x.text_to_df import TextToDf
from text_to_x.text_to import TextTo

class TextToSentiment(TextTo):
    def __init__(self, lang = None, method = "dictionary", type_token = None, 
                 detect_lang_fun = "polyglot", **kwargs):
        """
        lang (str): language code, if None language is detected using detect_lang_fun (which defaults to polyglot).
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return 
        method ("dictionary"|"bert"|fun): method used for sentiment analysis
        type_token (None|'lemma'|'token'): The type of token used. If None is chosen to be token automatically depending on method.
        'lemma' for dictionary otherwise 'token'. Only used if a tokenlist or a TextToDf is passed to texts_to_sentiment()
        """
        super().__init__(lang = lang, kwargs = kwargs, 
                         detect_lang_fun = detect_lang_fun)

        if type_token is None:
            self.type_token = 'lemma' if method == "dictionary" else 'token'

        if method == "dictionary":
            self.method = self.__get_sent_dict
        elif callable(method):
            self.method = method
        else: 
            raise ValueError(f"The chosen method: {self.method}")

    def texts_to_sentiment(self, texts):
        """
        texts (str|list|TextToDf): Should be a string, a list of strings or other iterable object or an object of class TextToDf
        """
        tokenlist = None
        assert isinstance(texts, (TextToDf, str, list)), \
            "'texts' must be str, list of strings or TextToDf."
        if isinstance(texts, TextToDf):
            tokenlist = [df[self.type_token] for df in texts.dfs]
            if self.lang is None:
                self.lang = texts.lang
            texts = texts.texts
        else:
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, list) and not isinstance(texts[0], str):
                # One may accidentally pass the list of preprocessed data frames
                raise TypeError(f"When 'texts' is a list, it must contain strings only.")
            self._detect_language(texts)
        return self.method(texts, tokenlist)
    
    def __get_sent_dict(self, texts, tokenlist):
        if isinstance(self.lang, str):
            self.__fetch_sent_lang(self.lang)
            if tokenlist is None:
                res = [self.__dict_model[self.lang](text) for text in texts]
            else:
                res = [self.__dict_model[self.lang](text, tokens) for text, tokens in zip(texts,tokenlist)]
            return pd.DataFrame(res)
        else:
            res = []
            loop_iter = zip(texts, self.lang) if tokenlist is None else zip(texts, self.lang, tokenlist)
            for t in loop_iter:
                if tokenlist is None:
                    text, l = t
                else:
                    text, l, tokens = t
                self.__fetch_sent_lang(l)
                df = self.__dict_model[l](text, tokens)
                df['lang'] = l
                res.append(self.__dict_model[l](text))
            return pd.DataFrame(res)
        
    def __fetch_sent_lang(self, lang):
        self.__dict_model = {}
        if lang in self.__dict_model:
            return None
        if lang == "en":
            analyser = Sentiment_en()
            self.__dict_model[lang] = analyser.polarity_scores
        elif lang == "da":
            analyser = Sentiment_da()
            self.__dict_model[lang] = analyser.polarity_scores
        else:
            raise ValueError("Language {lang} does not have a dictionary implemented")


# testing code
if __name__ == "__main__":
    import os
    os.getcwd()
    os.chdir("..")
    # make some data
    with open("test_data/fyrtårnet.txt", "r") as f:
        text = f.read()
        # just some splits som that the text aren't huge
    t1 = "\n".join([t for t in text.split("\n")[1:50] if t])
    t2 = "\n".join([t for t in text.split("\n")[50:100] if t])
    t3 = "\n".join([t for t in text.split("\n")[100:150] if t])

    # we will test it using a list but a single text will work as well
    texts = [t1, t2, t3]

    tts = TextToSentiment(lang = "da", method="dictionary")
    df = tts.texts_to_sentiment(texts)
    df

    # with TextToDf
    ttd = TextToDf()
    ttd.texts_to_dfs(texts)

    s = Sentiment_da()
    [s.polarity_scores(text = text, tokenlist=df['lemma']) for text, df in zip(texts, ttd.dfs)]
    tts = TextToSentiment(method="dictionary")
    df = tts.texts_to_sentiment(ttd)



