Background:
In this section, we provide some important definitions that help us understand how we obtain our vector embeddings.
Definition 1 (Abstract Syntax Tree): An AST for a function is a tuple <N, T, , s, , > where N is the set of non-terminal nodes, T is the set of terminal nodes,  is the set of values, s is the starting node, :N(NT)* is a function that maps non terminal nodes to its children nodes and :T is a function that maps a terminal node to a value.

Definition 2 (AST Path): An AST path p is a specific path in the AST from a terminal node n1 to another terminal node nk+1, where k is the length of the path. Specifically, an AST path of length k is a sequence {n1d1...nkdknk+1}, where niN, i [2, k] (i.e. non terminal) and di{,}, i[1,k] are movements up or down the tree.

Definition 3 (Path Context): A path context is a tuple <s, p, t> such that p is an AST path, and s=(n1), t=(nk+1). In other words, a path context represents a specific path in the AST by moving through the tree where we go from a value to another value.