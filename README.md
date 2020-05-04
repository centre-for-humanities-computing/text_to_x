# Text to X
You shouldn't text to your X but you should extract from text. Text To X, a quick an easy to use NLP pipeline for converting text to topics, tokens, sentiment and more.

---
## 🔧 Installation
Install by running the following line:
```
pip3 install git+https://github.com/centre-for-humanities-computing/text_to_x
```
To reinstall the package use the following code
```
pip3 install --force-reinstall --no-deps  git+https://github.com/centre-for-humanities-computing/text_to_x
```

---
## 📖 Tokenization and token tagging
Tokenization and token tagging utilized stanza, flair and keras for tokenization, lemmatization, pos-tagging, dependency parsing and NER-tagging.



### Example of use
Let's start of by loading some data, we will use the Danish "fyrtårnet" by HC. Andersen and use 3 shorts
splits of 50 sentences. For the full text set `short_splits=False`. It is also possible to set `data='origin_of_species'`.
```{python}
import text_to_x as ttx
texts = ttx.get_test_data(data="fyrtårnet", short_splits=True)
```

And the use is very simple:
```{python}
ttt = ttx.TextToTokens()
dfs = ttt.texts_to_tokens(texts)
```
```
Currently at text: 0
Use device: cpu
---
Loading: tokenize
With settings:
...
Currently at text: 1
Currently at text: 2
Currently at text: 3
```

Examining the output we see that dfs have a length equal to the number of strings in `texts` and that the output is a pandas dataframe.
``` {python}
len(dfs)
```
```
3
```
```
df = dfs[0] # take the first item
df.head()
```
```
n_sent        token        lemma  upos xpos dependency relation
0       0          Der          der   ADV    _                expl
1       0          kom        komme  VERB    _                root
2       0           en           en   DET    _                 det
3       0       soldat       soldat  NOUN    _                 obj
4       0  marcherende  marcherende  VERB    _               xcomp
```

In the more extended use cases the you can modify the arguments more e.g.:
```
ttt = ttx.TextToTokens(lang = ["da", "da", "da"],
                                tokenize="stanza",
                                lemmatize="stanza",
                                stemming=None,
                                pos="stanza",
                                mwt="stanza",
                                depparse="stanza",
                                ner="stanza")
dfs = ttt.texts_to_tokens(texts = texts)
```
Note that language can also be a list of languages and if left out the language is detected using polyglot.

---
## 🥳🤬 Sentiment Analysis 
Text to X utilized an altered version of a dictionary or a multilingual BERT. For the dictionary approach it used an altered version of [vaderSentiment](https://github.com/cjhutto/vaderSentiment) allowing for multiple languages and use of tokens, lemmas or stems derived from TextToTokens.

### Example of use
The simple use case is (using the same texts as above):
```
tts = ttx.TextToSentiment(lang = "da", method="dictionary")
df = tts.texts_to_sentiment(texts)
print(df)
```
```
neg    neu    pos  compound
0  0.060  0.851  0.089    0.9794
1  0.064  0.826  0.109    0.9973
2  0.031  0.780  0.189    0.9615
```

If we want to use it with TextToTokens we can do as follow:
```
# create the TextToTokens
ttt = ttx.TextToTokens()
ttt.texts_to_tokens(texts)

# initialize the TextToSentiment
tts = TextToSentiment(method="dictionary")

# simply pass the ttt as the first argument
df = tts.texts_to_sentiment(ttt)
```

---
## 🚧 Future improvements
In estimated order
- [x] Make a class TextToTopic for topic modelling using gensim mallet and LDA
- [x] add fast a fast tokenizer for TextToTokens
- [x] Add entity tagger
    - [x] add entity tagger for Danish
- [x] Update TextToDf to use Stanza instead of stanfordnlp
- [ ] Additions to the TextToSentiment class
    - [ ] add token_to_sentiment, which give the sentiment of each token
    - [ ] add sentence_to_sentiment, which give the sentiment of each sentence

---
## 🎓 References: 
>Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D. (2020). Stanza: A {Python} Natural Language Processing Toolkit for Many Human Languages. arXiv

>Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

> Lauridsen, G. A., Dalsgaard, J. A., & Svendsen, L. K. B. (2019). SENTIDA: A New Tool for Sentiment Analysis in Danish. Journal of Language Works-Sprogvidenskabeligt Studentertidsskrift, 4(1), 38-53.

