# Ideas
1. create abstract syntax trees from functions
2. Obtain paths of the AST
3. Preprocess dataset
4. use AST to create code vectors
5. Using directed acyclic graphs might improve model performance
6. Implementation of acyclic graphs vs AST

# changes
1. use only python codes
2. use transformers instead of LSTM

# preprocessing
1. generic variable names
2. generic function names


# TEsting ideas
1. use chat gpt to generate many similar functions, and evaluate the difference of vectors
2. Set a threshold with cosine similarity to check whether a prediction is "correct" or not.


dataset ->
split by functions, some are classes with many functions
turning code into ast tree
1. clean variable names
2. clean out function names, store function name as label
3. extract individual functions to put in dataset (instead of file being 1 row)
4. 90% of functions did not have comments
5. originally saved the comments, but decided not to use them
turns it back to a cleaned version of code

cleaned function goes into ast miner (limit the size TBD)
1. extracts  the paths, creates path contexts
2. output of astminer, reconstruct the rows from the output csv file. each examples and turn it into a str of path contexts
3. each row is a set of path contexts, every existing path contexts from the function
4. code2vec paper (steps taken)

model
2 models: encoder and decoder
encoder: model we generate, create a vector emb for the data
decoder: takes in vector emb, and tries to predict the function name (label)
both trained together
error is coming from output of the decoder (compared to label), and propogate back to encoder
that is how we train both models simultaneously.

cleaning var names
try to limit number of path contexts
using python instead of java
conv layer maybe, before encoder, so that it can select areas of