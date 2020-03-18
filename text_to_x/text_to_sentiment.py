"""
"""
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from text_to_sentiment.vaderSentiment.vaderSentiment_da import SentimentIntensityAnalyzer as SentimentIntensityAnalyzer_da
from text_to_sentiment.utils import detect_lang_polyglot


class TextToSentiment():
    def __init__(self, texts, detect_lang_fun = "polyglot", lang = None):
        """
        texts (str|list): Should be a string, a list or other iterable object
        lang (str): language code, if None language is detected using polyglot
        """
        self.__detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        self.sentiment_model = {}

        if isinstance(texts, str):
            texts = [texts]

        if isinstance(detect_lang_fun, str):
            detect_lang_fun = self.__detect_lang_fun_dict[detect_lang_fun]
        elif not callable(detect_lang_fun):
            raise ValueError(f"detect_lang_fun should be a string or callable not a {type(detect_lang_fun)}")
        
        if lang is None:
            self.lang = [detect_lang_polyglot(text)[0] for text in texts]
        else:
            self.lang = lang
                

        self.texts = texts

    def sentiment_score(self, method = "dictionary"):
        if method == "dictionary":
            return self.__get_sent_dict()
    
    def __get_sent_dict(self):
        if isinstance(self.lang, str):
            self.__fetch_sent_lang(self.lang)
            res = [self.sentiment_model[self.lang](text) for text in self.texts]
            return pd.DataFrame(res)
        else:
            res = [self.__get_sent_dict_inner(text, l) for text, l in zip(self.texts, self.lang)]
            return pd.DataFrame(res)
        

    def __get_sent_dict_inner(self, text, lang):
        self.__fetch_sent_lang(lang)
        res = self.sentiment_model[lang](text)
        res['lang'] = lang
        return res
    
    def __fetch_sent_lang(self, lang):
        if lang in self.sentiment_model:
            return None
        if lang == "en":
            analyser = SentimentIntensityAnalyzer()
            self.sentiment_model[lang] = analyser.polarity_scores
        elif lang == "da":
            analyser = SentimentIntensityAnalyzer(lexicon_file="vader_lexicon.txt")
            self.sentiment_model[lang] = analyser.polarity_scores
        else:
            raise ValueError("Language {lang} does not have a dictionary implemented")




# testing code
if __name__ == "__main__":
    # make some data
    with open("test_data/fyrtårnet.txt", "r") as f:
        text = f.read()
        # just some splits som that the text aren't huge
    t1 = "\n".join([t for t in text.split("\n")[1:50] if t])
    t2 = "\n".join([t for t in text.split("\n")[50:100] if t])
    t3 = "\n".join([t for t in text.split("\n")[100:150] if t])

    # we will test it using a list but a single text will work as well
    texts = [t1, t2, t3]

    tts = TextToSentiment(texts, lang = None)
    tts.sentiment_score()

df = pd.read_csv("vaderSentiment_NonEnglish/vader_lexicon_da.csv", encoding='ISO-8859-1')
d = df.set_index("stem")[ 'score'].to_dict()
