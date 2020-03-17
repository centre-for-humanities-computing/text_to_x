# Normalize text to dataframe
A pipeline for normalizing texts to dataframes.

# Installation
Install by running the following line:
```
pip3 install --force-reinstall --no-deps  git+https://github.com/centre-for-humanities-computing/normalize_text_to_df
```

# Example of use
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
import normalize_text_to_df as ttdf
dfs = ttdf.texts_to_dfs(texts)
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

The more extended use case is:
```
dfs = ttdf.texts_to_dfs(texts, lang = "da", method = "stanfordnlp", 
             args = {"processor":"tokenize,mwt,lemma,pos,depparse"})
```
Note that language can also be a list of languages.