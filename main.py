import jax
from jax import tree_util

test = {"item" : None, "hi" : [3, 2], "bit" : "4"}

leaves, treedef = tree_util.tree_flatten(test, lambda x : x is None)
leaves, treedef = tree_util.tree_flatten_with_path(test, lambda x : x is None)

print(leaves)
print(((1, 2), 3) == (1, 2))