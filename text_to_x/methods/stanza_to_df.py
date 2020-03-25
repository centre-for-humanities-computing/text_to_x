"""
"""
import os
from pathlib import Path

import pandas as pd
import stanza


def dl_missing_langs(langs, stanza_path):
    """
    downloads any missing languages from stanza

    Examples:
    >>> stanza_path = os.path.join(str(Path.home()), 'stanza_resources')
    >>> dl_missing_langs(langs = "da", stanza_path = stanza_path)
    """

    if isinstance(langs, str):
        langs = [langs]

    if stanza_path is not None and not os.path.exists(stanza_path):
        os.makedirs(stanza_path)

    dl_langs = [folder[:2] for folder in os.listdir(stanza_path)]
    for lang in langs:
        if lang not in dl_langs:
            try:
                stanza.download(lang, dir=stanza_path)
            except ValueError:
                raise ValueError(f"Language: '{lang}' does not exist in stanza.\
                                 Try specifying another language")


def stanza_to_df(texts, langs, stanza_path=None, silent=False, **kwargs):
    """
    lang (str|list)

    tokenize, pos-tag, dependency-parsing

    tokenize,mwt,lemma,pos,depparse


    Examples:
    >>> text = "Dette er en test text, den er skrevet af Kenneth Enevoldsen. \
        Mit telefonnummer er 12345678, og min email er notmymail@gmail.com"
    >>> stanza_to_df(text, langs = "da")
    >>> text = "My name is Kenneth Enevoldsen, i speak English and Danish."
    >>> stanza_to_df(text, langs = "en")
    """
    if isinstance(texts, str):
        texts = [texts]

    # Download missing SNLP resources for the detected/specified language
    if stanza_path is None:
        stanza_path = os.path.join(str(Path.home()), 'stanza_resources')
    dl_missing_langs(langs, stanza_path)

    if isinstance(langs, list):
        lang = langs[0]  # for dealing with multiple languages
    else:
        lang = langs

    res = []
    for i, text in enumerate(texts):
        if not silent:
            print(f"Currently at text: {i}")
        if i == 0:
            s_nlp = stanza.Pipeline(lang=lang, dir=stanza_path, **kwargs)
        elif isinstance(langs, list) and lang != langs[i]:
            lang = langs[i]
            s_nlp = stanza.Pipeline(lang=lang, dir=stanza_path, **kwargs)

        doc = s_nlp(text)

        sent_ids = dict()
        sent_n = None

        def __get_ent(n_sent, sent, word):
            nonlocal sent_ids
            nonlocal sent_n
            if sent_n != n_sent:
                sent_ids = {word.id: ent.type for ent in sent.ents
                            for word in ent.words}
            if word.id in sent_ids:
                return sent_ids[word.id]

        # extract from doc
        tmp = ((n_sent, word.text, word.lemma, word.upos, word.xpos,
                word.deprel, __get_ent(n_sent, sent, word))
               for n_sent, sent in enumerate(doc.sentences)
               for i, word in enumerate(sent.words))

        cols = ["n_sent", "token", "lemma", "upos", "xpos",
                "dependency relation", "ner"]
        df = pd.DataFrame(tmp, columns=cols)
        df['lang'] = lang
        res.append(df)
    return res


# for dep_edge in self.dependencies:
#             print((dep_edge[2].text, dep_edge[0].id, dep_edge[1]), file=file)
# testing code
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
