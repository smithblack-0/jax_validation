import jax
from jax.tree_util import tree_flatten, tree_unflatten, is_leaf, treedef_children, treedef_is_leaf, TreeDef

class W