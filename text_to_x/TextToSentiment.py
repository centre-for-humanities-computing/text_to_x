"""
"""
import pandas as pd

from transformers import pipeline

from text_to_x.vaderSentiment.vaderSentiment_en import (
    SentimentIntensityAnalyzer as Sentiment_en,
)
from text_to_x.vaderSentiment.vaderSentiment_da import (
    SentimentIntensityAnalyzer as Sentiment_da,
)
from text_to_x.vaderSentiment.vaderSentiment_se import (
    SentimentIntensityAnalyzer as Sentiment_se,
)
from text_to_x.vaderSentiment.vaderSentiment_no import (
    SentimentIntensityAnalyzer as Sentiment_no,
)
from text_to_x.TextToTokens import TextToTokens
from text_to_x.TextToX import TextToX


class TextToSentiment(TextToX):
    def __init__(
        self,
        lang=None,
        method="mult_bert",
        type_token=None,
        detect_lang_fun="polyglot",
        **kwargs,
    ):
        """
        lang (str): language code, if None language is detected using
        detect_lang_fun (which defaults to polyglot).
        detect_lang_fun (str|fun): fucntion to use for language detection.
        default is polyglot. But you can specify a user function, which return
        method ("dictionary"|"bert"|"danlp_bert_tone"|fun): method used for sentiment analysis
        type_token (None|'lemma'|'token'): The type of token used. If None is
        chosen to be token automatically depending on method.
        'lemma' for dictionary otherwise 'token'. Only used if a tokenlist or
        a TextToDf is passed to texts_to_sentiment()
        """
        super().__init__(lang=lang, kwargs=kwargs, detect_lang_fun=detect_lang_fun)

        if type_token is None:
            self.type_token = "lemma" if method == "dictionary" else "token"

        if method == "dictionary":
            self.method = self.__get_sent_dict
        elif method == "mult_bert":
            self.method = self.__get_sent_mult_bert
        elif method == "danlp_bert_tone":
            self.method = self.__get_sent_danlp_bert_tone

        elif callable(method):
            self.method = method
        else:
            raise ValueError(f"The chosen method: {self.method}")

    @staticmethod
    def __get_sent_danlp_bert_tone(texts, tokenlist):
        from danlp.models import load_bert_tone_model

        classifier = load_bert_tone_model()

        def get_proba(txt):
            res = classifier.predict_proba(txt)
            polarity, analytic = res
            pos, neu, neg = polarity
            obj, subj = analytic
            return pos, neu, neg, obj, subj

        return pd.DataFrame(
            [get_proba(txt) for txt in texts],
            columns=[
                "polarity_pos",
                "polarity_neu",
                "polarity_neg",
                "analytic_obj",
                "analytic_subj",
            ],
        )

    def texts_to_sentiment(self, texts):
        """
        texts (str|list|TextToDf): Should be a string, a list of strings or
        other iterable object or an object of class TextToDf
        """
        tokenlist = None
        assert isinstance(
            texts, (TextToTokens, str, list)
        ), "'texts' must be str, list of strings or TextToTokens object."
        if isinstance(texts, TextToTokens):
            tokenlist = [df[self.type_token] for df in texts.get_token_dfs()]
            if self.lang is None:
                self.lang = texts.lang
            texts = texts.texts
        else:
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, list) and not isinstance(texts[0], str):
                # One may accidentally pass the list of preprocessed data
                # frames
                raise TypeError(
                    f"When 'texts' is a list, it must contain \
                    strings only."
                )
            self._detect_language(texts)
        return self.method(texts=texts, tokenlist=tokenlist)

    def __get_sent_dict(self, texts, tokenlist):
        if isinstance(self.lang, str):
            self.__fetch_sent_lang(self.lang)
            if tokenlist is None:
                res = [self.__dict_model[self.lang](text) for text in texts]
            else:
                res = [
                    self.__dict_model[self.lang](text, tokens)
                    for text, tokens in zip(texts, tokenlist)
                ]
            return pd.DataFrame(res)
        else:
            res = []
            loop_iter = (
                zip(texts, self.lang)
                if tokenlist is None
                else zip(texts, self.lang, tokenlist)
            )

            for t in loop_iter:
                if tokenlist is None:
                    text, lang = t
                else:
                    text, lang, tokens = t
                self.__fetch_sent_lang(lang)
                df = self.__dict_model[lang](text, tokens)
                df["lang"] = lang
                res.append(self.__dict_model[lang](text))
            return pd.DataFrame(res)

    def __get_sent_mult_bert(self, texts, **kwargs):
        nlp = pipeline("sentiment-analysis")
        df = pd.DataFrame([nlp(t)[0] for t in texts])
        return df

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
        elif lang == "se":
            analyser = Sentiment_se()
            self.__dict_model[lang] = analyser.polarity_scores
        elif lang == "no":
            analyser = Sentiment_no()
            self.__dict_model[lang] = analyser.polarity_scores
        else:
            raise ValueError(
                "Language {lang} does not have a dictionary \
                implemented"
            )


# testing code
if __name__ == "__main__":
    # import os
    # os.getcwd()
    # os.chdir("..")

    from text_to_x.utils import get_test_data

    texts = get_test_data()

    tts = TextToSentiment(lang="da", method="dictionary")
    df = tts.texts_to_sentiment(texts)
    df

    tts = TextToSentiment(lang="da", method="mult_bert")
    df = tts.texts_to_sentiment(texts)
    df

    tts = TextToSentiment(lang="da", method="danlp_bert_tone")
    df = tts.texts_to_sentiment(texts)
    df

    # with TextToTokens
    ttt = TextToTokens()
    ttt.texts_to_tokens(texts)

    s = Sentiment_da()
    [
        s.polarity_scores(text=text, tokenlist=df["lemma"])
        for text, df in zip(texts, ttt.get_token_dfs())
    ]
    tts = TextToSentiment(method="dictionary")
    df = tts.texts_to_sentiment(ttt)
