# Text to X
An quick an easy to use NLP pipeline

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
## 📖 Tokenization and tagging
Normalization utilizes stanfordNLP for tokenization, lemmatization, pos-tagging, dependency parsing and Named entity recogntion (NER).


### Example of use
Let's start of by loading some data, we will use the Danish "fyrtårnet" by HC. Andersen
```{python}
with open("test_data/fyrtårnet.txt", "r") as f:
    text = f.read()
# No need to run the full text let's just split the text up
t1 = "\n".join([t for t in text.split("\n")[1:50] if t])
t2 = "\n".join([t for t in text.split("\n")[50:100] if t])
t3 = "\n".join([t for t in text.split("\n")[100:150] if t])

# we will test it using a list but a single text will work as well
texts = [t1, t2, t3]
```

And the use is very simple:
```{python}
import text_to_x as ttx
ttt = TextToTokens()
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

Examining the output we see
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
ttt = TextToTokens(lang = ["da", "da", "da"], method = "stanza", 
                args = {"processors":"tokenize,mwt,lemma,pos,depparse"})
dfs = ttt.texts_to_tokens(texts = texts)
```
Note that language can also be a list of languages and if left out the language is detected using polyglot.

---
## 🥳🤬 Sentiment Analysis 
Text to X utilized an altered version of a dictionary or a multilingual BERT (not yet implented). For the dictionary approach it used an altered version of [vaderSentiment](https://github.com/cjhutto/vaderSentiment) allowing for multiple languages and use of tokenization og lemmatization derived from TextToDf.

### Example of use
The simple use case is (using the same texts as above):
```
tts = TextToSentiment(lang = "da", method="dictionary")
df = tts.texts_to_sentiment(texts)
print(df)
```
```
neg    neu    pos  compound
0  0.060  0.851  0.089    0.9794
1  0.064  0.826  0.109    0.9973
2  0.031  0.780  0.189    0.9615
```

If we want to use it with TextToDf we can do as follow:
```
# create the TextToTokens
ttt = TextToTokens()
ttt.texts_to_tokens(texts)

# initialize the TextToSentiment
tts = TextToSentiment(method="dictionary")

# simply pass the ttt as the first argument
df = tts.texts_to_sentiment(ttt)
```

---
## 🚧 Future improvements
In estimated order
- [ ] Make a class TextToTopic for topic modelling using gensim mallet and LDA
- [ ] add fast a fast tokenizer for TextToTokens
- [x] Add entity tagger
    - [ ] add entity tagger for Danish
- [x] Update TextToDf to use Stanza instead of stanfordnlp
- [ ] Additions to the TextToSentiment class
    - [ ] add token_to_sentiment, which give the sentiment of each token
    - [ ] add sentence_to_sentiment, which give the sentiment of each sentence

---
## 🎓 References: 
>Qi, Peng and Zhang, Yuhao and Zhang, Yuhui and Bolton, Jason and Manning, Christopher D. (2020). Stanza: A {Python} Natural Language Processing Toolkit for Many Human Languages. arXiv

>Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

> Lauridsen, G. A., Dalsgaard, J. A., & Svendsen, L. K. B. (2019). SENTIDA: A New Tool for Sentiment Analysis in Danish. Journal of Language Works-Sprogvidenskabeligt Studentertidsskrift, 4(1), 38-53.

