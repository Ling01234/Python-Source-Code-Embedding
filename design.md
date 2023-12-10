# high level design document for our code to vector model

# dataset extraction and design
#   1. use an existing open source collection of python functions (they are annotated with descriptions)
#   2. clean: remove function and variable names. We want our model to learn the semantics of the code, not the chosen names.
#   3. output: a set of cleaned functions as multi-line strings. Each function will have an accompanying annotation

# code to AST tree
#   1. convert multi-line function string into AST tree

# Transformer Model
1. Figure out our vocab and vocab size, number of unique tokens
2. create an embedding layer. we want to use this to map each token to a dense vector representation.
3. create a transformer encoder, which consists of many feedforward NN.
4. create a transformer output layer. since we want a fixed-size vector representation of the code, the size is the size of the vocab (?)
5.