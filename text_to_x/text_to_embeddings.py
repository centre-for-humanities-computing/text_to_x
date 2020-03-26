from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg 

import umap
    
    
class SvdEmbeddings():
    '''
    train and operate with PMI-SVD embeddings
    
    Examples:
    - training with a manually set class instance
        your_instance = SvdEmbeddings(texts)
        your_instance.train(front_window = int, back_window = int, embedding_dim = int)
    
    - training with default settings:
        your_instance = svd2vec_run_default(texts)
        
    - visualizing the fit
    '''
    
    def __init__(self, texts, texts_colname = 'lemma'):
        '''
        texts (iterable | TextToToken): input list of dataframes acquired from text_to_x.text_to_token. The "lemma" column is used. 
        texts_colname (string): column from TextToToken DataFrame to process ("lemma" / "stem" / "word")
        '''

        self.model = {}
        self.tok2ind = {}
        self.ind2tok = {}
        self.unigram_counts = {}
        
        # check if input is TextToToken or simialr
        if not isinstance(texts[0], pd.DataFrame):
            raise ValueError("Input texts are not in a DataFrame or a list of DataFrames, please use TextToToken for preprocessing")
        
        # unpack lemmas 
        lemmas_to_list = []
        for doc in texts:
            doc_to_list = doc.loc[:, texts_colname].tolist()
            lemmas_to_list.append(doc_to_list)

        self.texts = lemmas_to_list
      
    
    def train(self, back_window, front_window, embedding_dim):
        '''
        train svd embeddings from texts
        
        - back_window (int): max number of words to look behind in a skipgram 
        - front_window (int):, max number of words to look forward in a skipgram 
        - embedding_dim (int): number of dimensions to fit the embeddings to
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
        alpha = 0.75
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
        
        Example: 
        In: self.similarity('hpv', 'vaccine')
        
        Out: 0.11681473580251966
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
        
        Example:
        In: self.find('hpv')
        
        Out: array([ 0.00435369, ...,  0.332069  ])
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
        
        Example:
        In: self.similar_to_query('hpv')
        
        Out: [(0.3425330655678359, 'type'),
             (0.29492709699932357, 'eksempel'),
             (0.2840084462156332, 'larynxcance'),
             (0.2736081170128576, 'sygdom'),
             (0.24797964395624894, 'forårsag')]
        '''
        
        if not isinstance(query, str):
            raise ValueError(f"query expected a string")
            
        if not isinstance(n_similar, int):
            raise ValueError(f"n_similar expected int")
            
        
        vec = self.find(query)
        
        # cosine similarity
        cos_sim_matrix = np.dot(self.model, vec)/(np.linalg.norm(self.model)*np.linalg.norm(vec))
        
        list_similar = []

        for i in np.argpartition(-1 * cos_sim_matrix, n_similar + 1)[:n_similar + 1]:

            if self.ind2tok[i] == query:
                continue

            list_similar.append((float(cos_sim_matrix[i]), self.ind2tok[i]))

        return sorted(list_similar, reverse=True)
    
    
    def similar_to_vec(self, vec, n_similar = 5):
        '''
        get most similar trained embeddings to an input vector
        cosine similarity is the metric used (min = -1, max = 1)
        
        - vec: np.array, input vector (also a result of operation with vectors)
        - n_similar: int, number of most simialr words to output
        
        Example:
        In: self.similar_to_vec(vec = self.find('hpv') - self.find('pige'))
        
        Out: [(1.1030897070723877, 'hpv'),
             (0.3426745031741564, 'type'),
             (0.2964467662004768, 'eksempel'),
             (0.2854145615758435, 'vort'),
             (0.285052615742971, 'larynxcance'),
             (0.27461291543657906, 'sygdom')]
        '''
        
        # cosine similarity
        cos_sim_matrix = np.dot(self.model, vec)/(np.linalg.norm(self.model)*np.linalg.norm(vec))
        
        list_similar = []

        for i in np.argpartition(-1 * cos_sim_matrix, n_similar + 1)[:n_similar + 1]:

            list_similar.append((float(cos_sim_matrix[i]), self.ind2tok[i]))

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
        
        self.model2d = reduced.fit_transform(self.model)  
    
    
def svd2vec_small_dataset(texts):
    svd = SvdEmbeddings(texts = texts)
    svd.train(back_window=2, front_window=2, embedding_dim=50)
    return svd  

def svd2vec_run_default(texts):
    svd = SvdEmbeddings(texts = texts)
    svd.train(back_window=5, front_window=5, embedding_dim=300)
    return svd 
