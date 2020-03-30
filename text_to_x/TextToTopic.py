"""
"""
import gensim
import pyLDAvis

from text_to_x.textToX import TextToX
from text_to_x.TextToTokens import TextToTokens


class TextToTopic(TextToX):
    def __init__(self,
                 docs,
                 tokentype="lemma",
                 stopword_removal="nltk",
                 pos_tags="",
                 lang=None,
                 detect_lang_fun='polyglot',
                 **kwargs):
        """
        docs (list, TextToTokens) list of string or object of type text to
        tokens.
        tokentype ('lemma'|'stem'|'token'): What token type should be used
        by default they are simplified to lemma, but can also be stem or
        token
        """
        super().__init__(lang=lang, kwargs=kwargs,
                         detect_lang_fun=detect_lang_fun)

        if isinstance(docs, list):
            self.ttt = TextToTokens(lang=self.lang, **kwargs)
        elif isinstance(docs, TextToTokens):
            self.ttt = docs
        else:
            ValueError(f"docs should be a list of string or a object of \
                         type TextToTokens, not a type {type(docs)}")

        if tokentype is None:
            tokentype = 'token'
        if tokentype:
            self.tokenlist = [df[tokentype] for df in self.ttt.__dfs]

    def train_topic(self):
        pass

# visualize
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
