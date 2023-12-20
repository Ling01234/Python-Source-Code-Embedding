# Learning of Python Source Code From Structure and Context
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Project Description
In this paper, we introduce a neural model designed to encapsulate the semantics of Python
functions and to predict function names based
on code snippets. The core idea revolves
around representing a code snippet as a fixed
length vector embedding. This is achieved
through a process involving the decomposition
of code into an abstract syntax tree, extraction
of path contexts, and feeding this set of path
contexts to our encoder transformer. Then, the
decoder processes this vector embedding and
predicts a method name.

For instance, the two Python functions below should have nearly identical vector embeddings.
```python
def f(x):
    x = x + 10
    x = x**2
    return x
```
```python
def g(y):
    y = (y + 10)**2
    return y
```

# Dataset
In this paper, we used Hugging Face's ``the-stack-smol`` dataset. This dataset is a small subset (approx. 0.1%) of
``the-stack`` dataset, where each programming language has 10 000 random samples
from the original dataset.

# High Level Ideas
On a very high level, our pipeline is as follows:
1. Preprocesses the each Python file and extracts all functions. Each function will be an individual sample in our dataset.
2. Clean out the function by generalizing the variable and method names, and removing comments.
3. Use the cleaned code to generate the set of context paths.
4. Pass the context paths into the transformer model. The encoder outputs a fixed length vector embedding; the decoder uses this vector to predict a method name.

# Links
**📖 [Documentation]**
&ensp;|&ensp;
[ProQuest Document](https://escholarship.org/content/qt5qx4b1xh/qt5qx4b1xh.pdf?t=qviese)
&ensp;|&ensp;
[Code2Vec Code](https://code2vec.org/)
&ensp;|&ensp;
[Word2Vec Summary](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)

