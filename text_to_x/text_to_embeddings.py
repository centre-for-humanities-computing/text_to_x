"""
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg 
import umap

    
class SvdEmbeddings():
    '''
    train and operate with PMI-SVD embeddings
    
    Usage:
    Run preprocessing,
    >>> from text_to_x.text_to_tokens import TextToTokens
    >>> from text_to_x.utils import get_test_data
    >>> raw_text = get_test_data()
    >>> ttt = TextToTokens()
    >>> sample_text = ttt.texts_to_tokens(raw_text)
    
    (option 1) then use one of the convenience functions, 
    >>> SvdModel1 = svd2vec_run_default(sample_text)
    >>> SvdModel2 = svd2vec_small_dataset(sample_text)
    
    (option 2) or make a class instance
    >>> SvdInst = SvdEmbeddings(sample_text)
    >>> SvdInst.train(front_window = 2, back_window = 2, embedding_dim = 50)
    
    Model is a numpy array, where rows are words and columns embedding dimensions 
    >>> isinstance(SvdInst.model, np.ndarray)
    True
    
    In this model, we have 460 words over 50 dimnesions
    >>> SvdInst.model.shape
    (460, 50)
    
    Words present in the model
    >>> isinstance(SvdInst.unigram_counts, pd.DataFrame)
    True
    
    Find vector representation 
    >>> SvdInst.find('soldat')
    array([ 0.05489066, -0.06184785,  0.02155402,  0.07336988,  0.15562366,
           -0.05586125, -0.25812332,  0.02042852,  0.05166734,  0.03197265,
            0.09791423, -0.11929871, -0.0229645 , -0.06041331, -0.04729205,
           -0.06330125, -0.05803931, -0.03184649, -0.1423933 ,  0.09108703,
            0.16427856, -0.05856962,  0.12271073, -0.13391868, -0.12081928,
           -0.00627897, -0.07468613, -0.04102333,  0.17715846,  0.03979193,
            0.00880396, -0.09238536,  0.10664236, -0.00629155,  0.22754591,
           -0.2665231 , -0.09757153,  0.09906008, -0.00108509, -0.06677161,
           -0.01872435,  0.05611655, -0.09478669,  0.04123027, -0.05015442,
           -0.15494153, -0.11920451, -0.11169791,  0.00589421, -0.16254055])
    
    How similar are 'soldat' and 'hund'
    >>> SvdInst.similarity('soldat', 'hund')
    -0.03461061459147635
    
    Find most similar words to 'soldat'
    >>> SvdInst.similar_to_query('hund')
    [(0.030169664949560287, 'komme'), (0.02888018261193621, 'der'), (0.026827027447386905, 'igen'), (0.02651235798431899, 'sidde'), (0.026125431437429042, 'danse')]
    
    Find most similar words to 'hund' minus 'soldat'
    >>> SvdInst.similar_to_vec(SvdInst.find('hund') - SvdInst.find('soldat'))
    [(0.04576068129552308, 'hund'), (0.024899982533863614, 'der'), (0.024181414732223933, 'tre'), (0.02334725064323222, 'to'), (0.021627587366789172, 'komme'), (0.020995251865100963, 'øje')]
    
    Reduce semantic space dimensions to two components
    >>> SvdInst.reduce_dim_umap()
    >>> SvdInst.modelumap.shape
    (460, 2)
    '''
    
    def __init__(self, texts, texts_colname = 'lemma'):
        '''
        texts (iterable | TextToToken): input list of dataframes acquired from text_to_x.text_to_token. The "lemma" column is used by default. 
        texts_colname (str): column from TextToToken DataFrame to process ("lemma" / "stem" / "word")
        '''

        self.model = {}
        self.tok2ind = {}
        self.ind2tok = {}
        self.unigram_counts = {}
        
        # check if input is TextToToken or simialr
        if isinstance(texts, list):
            if not isinstance(texts[0], pd.DataFrame):
                raise ValueError(
                    "Input texts are not in a DataFrame or a list of DataFrames, please use TextToToken for preprocessing"
                )
        
            # unpack lemmas 
            lemmas_to_list = [doc.loc[:, texts_colname].tolist() for doc in texts]
            
        # in case input is only one doc
        if isinstance(texts, pd.DataFrame):
            lemmas_to_list = texts.loc[:, texts_colname].tolist()

        self.texts = lemmas_to_list
      
    
    def train(self, back_window, front_window, embedding_dim, alpha = 0.75):
        '''
        train svd embeddings from texts
        
        - back_window (int): max number of words to look behind in a skipgram 
        - front_window (int):, max number of words to look forward in a skipgram 
        - embedding_dim (int): number of dimensions to fit the embeddings to
        - alpha (float in [0,1]): context distribution smoothing factor. Coocurrance counts will be raised to the power of alpha. alpha = 1 means no smoothing. The lowering the number decreases the PMI of a word appearing in rare contexts. See Levy, Goldberg & Dagan (2015) for details.
        '''
        
        ### UNIGRAMS
        tok2indx = dict()
        unigram_counts = Counter()
        for ii, text in enumerate(self.texts):
            for token in text:
                unigram_counts[token] += 1
                if token not in tok2indx:
                    tok2indx[token] = len(tok2indx)
        indx2tok = {indx: tok for tok, indx in tok2indx.items()}

    
        ### SKIPGRAMS
        skipgram_counts = Counter()
        for itext, text in enumerate(self.texts):
            for ifw, fw in enumerate(text):
                icw_min = max(0, ifw - back_window)
                icw_max = min(len(text) - 1, ifw + front_window)
                icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
                for icw in icws:
                    skipgram = (text[ifw], text[icw])
                    skipgram_counts[skipgram] += 1    

    
        ### COVARIANCE MATRIX
        row_indxs = list()
        col_indxs = list()
        dat_values = list()
        for (tok1, tok2), sg_count in skipgram_counts.items():
            tok1_indx = tok2indx[tok1]
            tok2_indx = tok2indx[tok2]        
            row_indxs.append(tok1_indx)
            col_indxs.append(tok2_indx)
            dat_values.append(sg_count)
        wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))


        ### POINTWISE MUTUAL INFORMATION
        # smoothed pmi (spmi), smoothed positive pmi (sspmi)
        num_skipgrams = wwcnt_mat.sum()
        assert(sum(skipgram_counts.values())==num_skipgrams)
        row_indxs = []
        col_indxs = []

        spmi_dat_values = []
        sppmi_dat_values = []

        # smoothing factor
        nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)
        sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
        sum_over_words_alpha = sum_over_words**alpha
        sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

        for (tok1, tok2), sg_count in skipgram_counts.items():
            tok1_indx = tok2indx[tok1]
            tok2_indx = tok2indx[tok2]

            nwc = sg_count
            Pwc = nwc / num_skipgrams
            nw = sum_over_contexts[tok1_indx]
            Pw = nw / num_skipgrams

            nca = sum_over_words_alpha[tok2_indx]
            Pca = nca / nca_denom

            spmi = np.log2(Pwc/(Pw*Pca))
            sppmi = max(spmi, 0)

            row_indxs.append(tok1_indx)
            col_indxs.append(tok2_indx)
            spmi_dat_values.append(spmi)
            sppmi_dat_values.append(sppmi)

        spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
        sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))


        # SVD of word vectors
        uu, ss, vv = linalg.svds(sppmi_mat, embedding_dim)

        word_vecs = uu + vv.T


        # unigram counts df
        counter_df = (pd.Series(unigram_counts, name = 'count')
                      .reset_index()
                      .rename(columns={'index': 'token'})
                      .reset_index()
                      .sort_values(by='count', ascending=False)
                     )
        
        # save stuff
        self.model = word_vecs
        self.tok2ind = pd.Series(tok2indx)
        self.ind2tok = pd.Series(indx2tok)
        self.unigram_counts = counter_df
    
    
    ###
    ### EMBEDDING OPERATIONS
    ###
    
    def similarity(self, word1, word2):
        '''
        calcualte the dot product between two given embeddings. Order doesn't matter.
        - word1: str, first word to query
        - wrod2: str, second word to query
        '''
        
        if not all(isinstance(i, str) for i in [word1, word2]):
            raise ValueError(f"expected string, please input a string to query")
        
        if word1 in self.tok2ind and word2 in self.tok2ind:
            index_word1 = self.tok2ind[word1]
            index_word2 = self.tok2ind[word2]
            return np.dot(self.model[index_word1], self.model[index_word2])
        
        else:
            if word1 in self.tok2ind and word2 not in self.tok2ind:
                raise ValueError(f"{word2}: query not in dataset")
            elif word1 not in self.tok2ind and word2 in self.tok2ind:
                raise ValueError(f"{word1}: query not in dataset")
                
                
    def find(self, query):
        '''
        input word, get vector representation
        '''
        if query in self.tok2ind:
            query_index = self.tok2ind[query]
            return self.model[query_index]
        
        raise ValueError("the word you're querying is not present in the model")
        
                
                
    def similar_to_query(self, query, n_similar = 5):
        '''
        print n most similar words to query (default 5)
        cosine similarity is the metric used (min = -1, max = 1)
        
        - query: str, word to query
        - n_similar: int, number of most simialr words to output
        '''
        
        if not isinstance(query, str):
            raise ValueError(f"query expected a string")
            
        if not isinstance(n_similar, int):
            raise ValueError(f"n_similar expected int")
            
        
        vec = self.find(query)
        
        # cosine similarity
        cos_sim_matrix = np.dot(self.model, vec)/(np.linalg.norm(self.model)*np.linalg.norm(vec))

        list_similar = [(float(cos_sim_matrix[i]), self.ind2tok[i]) for i in 
                np.argpartition(-1 * cos_sim_matrix, n_similar + 1)[:n_similar + 1] 
                if self.ind2tok[i] != query]

        return sorted(list_similar, reverse=True)
    
    
    def similar_to_vec(self, vec, n_similar = 5):
        '''
        get most similar trained embeddings to an input vector
        cosine similarity is the metric used (min = -1, max = 1)
        
        - vec: np.array, input vector (also a result of operation with vectors)
        - n_similar: int, number of most simialr words to output
        '''
        
        # cosine similarity
        cos_sim_matrix = np.dot(self.model, vec)/(np.linalg.norm(self.model)*np.linalg.norm(vec))
        
        list_similar = [(float(cos_sim_matrix[i]), self.ind2tok[i]) for i in
                np.argpartition(-1 * cos_sim_matrix, n_similar + 1)[:n_similar + 1]]

        return sorted(list_similar, reverse=True)
    
    
    def reduce_dim_umap(self):
        '''
        reduce the dimensions of tranined embeddings using UMAP. Output is 2D for easy visualization.
        Algorithm parameters are pre-defined here to give a drity overview of the global structure of the word-embedding model.
        Results may vary quite a bit based on UMAP paramenters. 
        
        For tweaking the parameters, see https://umap-learn.readthedocs.io/en/latest/parameters.html
        '''
        
        reduced = umap.UMAP(
            #reduce to 2 dimensions
            n_components = 2,
            #preserve local structure (low number) or global structure (high number) of the data
            n_neighbors = 15,
            #how tightly do we alow to pack points together
            min_dist = 0.1, 
            #correlational metric
            metric = 'cosine')
        
        self.modelumap = reduced.fit_transform(self.model)  
    
    
def svd2vec_small_dataset(texts):
    svd = SvdEmbeddings(texts = texts)
    svd.train(back_window=2, front_window=2, embedding_dim=50)
    return svd  

def svd2vec_run_default(texts):
    svd = SvdEmbeddings(texts = texts)
    svd.train(back_window=5, front_window=5, embedding_dim=300)
    return svd 

if __name__ == "__main__":
    # run tests
    import doctest
    doctest.testmod(verbose=True)
    