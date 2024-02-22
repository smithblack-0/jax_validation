import jax
import textwrap
from jax import tree_util
from typing import Tuple, Optional, Union, Any, List, Dict, Sequence
from .patching import Validator



class Schema:
    """
    A class designed to represent the structure
    of a pytree for downstream validation
    features.

    What is a Schema?
    ================

    A schema contains a jax pytree, and
    said pytree may be extended. The schema,
    however, must end ONLY in leaves that
    are either Validator, or None

    What is a Schema's Structure?
    =============================

    The structure of a schema is the branches of
    the tree, excluding the contents of the leafs

    What are the leafs of a schema?
    ===============================

    A schema can have any mix of 'Validator' or 'None'
    leaves. A 'Validator' leaf will indicate that this
    particular validator should be applied at this location
    or broadcast across this location on any target
    tree that might come across the schema.

    What does it mean for a tree to be compatible with a schema?
    ============================================================

    For a PyTree to be compatible with a schema, it must be the
    case that:

    1) The PyTree has structure that is the same as, or broadcastable
       with, the schema
    2) At every leaf in the schema that had a Validator, calling that validator
       against the corresponding leaf -or leaves if broadcasting -


    What is Schema Broadcasting?
    ============================

    Tensor broadcasting allows one feature to condition
    many separate dimensions, simply by replicating the result
    for each dimensions.

    Schema broadcasting does something very similar, but for
    tree structure. If a tree Structure defined in the schema





    Schemas support pytree broadcasting by
    prefix tree, much like jax tree mapping
    does.
    """

    @staticmethod
    def is_valid_leaf(leaf: Any):
        if not isinstance(leaf, Validator) and leaf is not None:
            msg = f"""
            An invalid leaf was detected while defining a schema

            The leaves of the pytree were expected to only consist of None or validator, 
            but a leaf was found of type {type(leaf)}
            """
            msg = textwrap.dedent(msg)
            raise TypeError(msg)



    def __init__(self, schema: Any):
        tree_util.tree_map(self.is_valid_leaf, schema)
        self.schema = schema