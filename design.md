# high level design document for our code to vector model

# dataset extraction and design
#   1. use an existing open source collection of python functions (they are annotated with descriptions)
#   2. clean: remove function and variable names. We want our model to learn the semantics of the code, not the chosen names.
#   3. output: a set of cleaned functions as multi-line strings. Each function will have an accompanying annotation

# code to AST tree
#   1. convert multi-line function string into AST tree