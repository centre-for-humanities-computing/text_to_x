"""

"""

# Imports ...
from text_to_x.text_to_df import TextToDf
from text_to_x.text_to_sentiment import TextToSentiment
from text_to_x.utils import detect_lang_polyglot


class Texts():
    def __init__(self, texts, 
                 language = None, 
                 preprocess_method = "stanfordnlp", 
                 preprocessors = ["tokenize", "mwt", "lemma", "pos", "depparse"], 
                 detect_lang_fun = "polyglot", 
                 **kwargs):
        """
        texts (list): texts to process.
        language (str): language code(s), if None language is detected using detect_lang_fun (which defaults to polyglot). Can be list of codes.
        detect_lang_fun (str|fun): fucntion to use for language detection. default is polyglot. But you can specify a user function, which return 
        preprocess_method (str|fun): method used for normalization
        preprocessors (list): names of processes to apply in the preprocessing stage
        """

        self.raw_texts = texts
        self.__kwargs = kwargs
        self.language = language
        self.__preprocess_method = preprocess_method
        self.preprocessors = preprocessors
        self.__preprocessor_args = {"processor": ",".join(self.preprocessors)}
        self.__detect_lang_fun = detect_lang_fun
        self.__sentiment_scores = None
        
        if self.language is None:
            self.language = self.__detect_languages(detect_lang_fun)

        self.preprocessed_texts = self.__preprocess()
        
    def __detect_languages(self, detect_lang_fun):
        detect_lang_fun_dict = {"polyglot": detect_lang_polyglot}
        if isinstance(detect_lang_fun, str):
            detect_lang_fun = detect_lang_fun_dict[detect_lang_fun]
        elif not callable(detect_lang_fun):
            raise TypeError(f"detect_lang_fun should be a string or callable not a {type(detect_lang_fun)}")
        return [detect_lang_fun(text, **self.__kwargs) for text in self.raw_texts]

    def __preprocess(self):
        ttd = TextToDf(lang = self.language, 
                       method = self.__preprocess_method, 
                       args = self.__preprocessor_args)
        return ttd.texts_to_dfs(texts = self.raw_texts)

    def score_sentiment(self, method = "dictionary", type_token = None):
        """
        method ("dictionary"|"bert"|fun): method used for sentiment analysis
        type_token (None|'lemma'|'token'): The type of token used. If None is chosen to be token automatically depending on method.
          'lemma' for dictionary otherwise 'token'.

        Use get_sentiments() method to extract scores.
        """
        tts = TextToSentiment(lang=self.language, method=method, type_token=type_token)
        self.__sentiment_scores = tts.texts_to_sentiment(self.preprocessed_texts)

    def get_sentiments(self):
        if self.__sentiment_scores is None:
            raise RuntimeError("The score_sentiment() method has not been called yet.")
        return self.__sentiment_scores

    def model_topics(self):
        pass

    def get_topics(self):
        pass



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

    # Init Texts object
    tt = Texts(texts, languages = "da")
    print(tt.preprocessed_texts)
    # Score sentiment
    tt.score_sentiment()
    print(tt.get_sentiments())
    # Topic modeling
    # tt.model_topics()


    


    
