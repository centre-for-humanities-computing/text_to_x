"""
"""
import gensim
import pyLDAvis

from text_to_x.text_to_x import TextToX


class TextToTokens(TextToX):
    pass

# visualize
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)