"""
"""
import os

import pandas as pd

import stanfordnlp

def dl_missing_langs_snlp(langs, stanfordnlp_path):
    """
    downloads any missing languages from stanford NLP resources

    Examples:
    >>> dl_missing_langs_snlp(langs = "da", stanfordnlp_path = os.getcwd() + "/stanfordnlp_resources")
    """


    if isinstance(langs, str):
      langs = [langs]
    
    if not os.path.exists(stanfordnlp_path):
      os.makedirs(stanfordnlp_path)

    dl_langs = [folder[:2] for folder in os.listdir(stanfordnlp_path)]
    for lang in langs:
        if lang not in dl_langs:
          try:
              stanfordnlp.download(lang, resource_dir=stanfordnlp_path, force = True)
          except ValueError:
            ValueError(f"Language: '{lang}' does not exist in stanford NLP. Try specifying another language")




def stanfordnlp_to_df(texts, langs, stanfordnlp_path = None, silent = False, **kwargs):
    """
    lang (str|list)

    tokenize, pos-tag, dependency-parsing

    tokenize,mwt,lemma,pos,depparse


    Examples:
    >>> text = "Dette er en test text, den er skrevet af Kenneth Enevoldsen. Mit telefonnummer er 12345678, og min email er notmymail@gmail.com"
    >>> snlp_to_df(text, lang = "da")
    """
    if isinstance(texts, str):
        texts = [texts]

    # Download missing SNLP resources for the detected/specified language
    if stanfordnlp_path == None:
      stanfordnlp_path = os.getcwd() + "/stanfordnlp_resources"
    dl_missing_langs_snlp(langs, stanfordnlp_path)
    
    if isinstance(langs, list):
        lang = langs[0] # for dealing with multiple languages
    else:
      lang = langs

    res = []
    for i, text in enumerate(texts):
        if not silent:
          print(f"Currently at text: {i}")
        if i == 0:
            s_nlp = stanfordnlp.Pipeline(lang = lang, models_dir = stanfordnlp_path, **kwargs)
        elif isinstance(langs, list) and lang != langs[i]:
            lang = langs[i]
            s_nlp = stanfordnlp.Pipeline(lang = lang, models_dir = stanfordnlp_path, **kwargs)
        
        doc = s_nlp(text)
        
        # extract from doc
        l = ( (n_sent, word.text, word.lemma, word.upos, word.xpos, word.dependency_relation) for n_sent, sent in enumerate(doc.sentences) for word in sent.words)
        df = pd.DataFrame(l, columns = ["n_sent", "token", "lemma", "upos", "xpos", "dependency relation"])
        res.append(df)
    return res