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
    if stanza_path is None:
        stanza_path = os.path.join(str(Path.home()), 'stanza_resources')

    if isinstance(langs, str):
        langs = [langs]
    if not os.path.exists(stanza_path):
        os.makedirs(stanza_path)
    dl_langs = {folder[:2] for folder in os.listdir(stanza_path)}
    for lang in langs:
        if lang not in dl_langs:
            try:
                stanza.download(lang, dir=stanza_path)
            except Exception:
                raise Exception(f"Language: '{lang}' is not supported by stanza.\
                                 Try specifying another language")


def stanza_to_df(texts,
                 langs,
                 stanza_path=os.path.join(str(Path.home()),
                                          'stanza_resources'),
                 verbose=True, **kwargs):
    """
    lang (str|list)

    tokenize, pos-tag, dependency-parsing

    tokenize,mwt,lemma,pos,depparse


    Examples:
    >>> text = "Dette er en test text, den er skrevet af Kenneth Enevoldsen. \
        Mit telefonnummer er 12345678, og min email er notmymail@gmail.com"
    >>> dfs = stanza_to_df(text, langs = "da", verbose = False)
    >>> text = "My name is Kenneth Enevoldsen, i speak English and Danish."
    >>> dfs = stanza_to_df(text, langs = "en", verbose = False)
    """
    if isinstance(texts, str):
        texts = [texts]

    dl_missing_langs(langs, stanza_path)

    if isinstance(langs, list):
        lang = langs[0]  # for dealing with multiple languages
    else:
        lang = langs

    res = []
    for i, text in enumerate(texts):
        if verbose:
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
               for word in sent.words)

        cols = ["n_sent", "token", "lemma", "upos", "xpos",
                "dependency relation", "ner"]
        df = pd.DataFrame(tmp, columns=cols)
        res.append(df)
    return res


def stanza_gen(texts,
               lang,
               processors="tokenize,mwt,lemma,pos,depparse,ner",
               stanza_path=os.path.join(str(Path.home()),
                                        'stanza_resources'),
               verbose=True,
               **kwargs):
    """
    texts (iter): an iterator of strings
    lang (str): language code
    stanza_path (path): the path for saving stanza resources

    Examples:
    >>> sg = stanza_gen(texts = ["dette er en test text"], lang = "da",\
        verbose=False)
    >>> type(sg)
    <class 'generator'>
    >>> sg_unpacked = list(sg)
    >>> type(sg_unpacked[0])
    <class 'pandas.core.frame.DataFrame'>
    """
    s_nlp = stanza.Pipeline(lang=lang, processors=processors,
                            dir=stanza_path, verbose=verbose, **kwargs)
    for text in texts:
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
        extr = ((n_sent,  # sentence number
                word.text,
                word.lemma,
                word.upos, word.xpos,  # pos-tags
                word.deprel, __get_ent(n_sent, sent, word))
                for n_sent, sent in enumerate(doc.sentences)
                for word in sent.words)
        cols = ["n_sent", "token", "lemma", "upos", "xpos",
                "dependency relation", "ner"]
        yield pd.DataFrame(extr, columns=cols)


# testing code
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
