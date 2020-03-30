
import pandas as pd

from text_to_x.TextToTokens import TextToTokens


def extract_concordance(preprocessed_texts, tokens, type_token="token",
                        sentence_id="n_sent", lower=True):
    """
    For a set of tokens, extract the sentences they occur in.

    preprocessed_texts (list of data frames | TextToTokens): Data frames with
    tokens.
    tokens (list of str): List of tokens to find concordances for.
    type_token (str): Either 'token', 'lemma', or 'stem'.
    sentence_id (str): Name of column with sentence identifier.
    lower (bool): Whether to match the tokens in lowercase.
    Returned sentences will have original case.
    """
    if isinstance(preprocessed_texts, TextToTokens):
        preprocessed_texts = preprocessed_texts.get_token_dfs()
    elif isinstance(preprocessed_texts, list) and not \
            isinstance(preprocessed_texts[0], pd.DataFrame):
        raise TypeError("When preprocessed_texts is a list, it must contain \
            data frames.")
    assert type_token in ["token", "lemma", "stem"], \
        "type_token must be one of 'token', 'lemma', and 'stem'."

    if lower:
        tokens = [tok.lower() for tok in tokens]

    def extract_concordance_single(df):
        tmp_token_name = "concordance_token"
        df[tmp_token_name] = df[type_token]

        if lower:
            df[tmp_token_name] = [tok.lower() for tok in df[tmp_token_name]]

        df_matches = df[df[tmp_token_name].isin(tokens)]

        def create_concordance(tok):
            concordance_sentences = df_matches[df_matches[tmp_token_name].
                                               isin(tokens)][sentence_id].\
                    unique()
            df_token_sentences = df[df[sentence_id].
                                    isin(concordance_sentences)]
            concordances = df_token_sentences.\
                groupby([sentence_id])[type_token].\
                apply(list).\
                reset_index()
            concordances.columns = [sentence_id, "sentence"]
            concordances[type_token] = tok
            return concordances

        concordances_df = pd.concat([create_concordance(tok)
                                     for tok in tokens])
        concordances_df = concordances_df[[sentence_id, type_token,
                                           "sentence"]]
        return concordances_df

    return [extract_concordance_single(df) for df in preprocessed_texts]
