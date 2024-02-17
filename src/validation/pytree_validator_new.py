import jax
import textwrap
from jax import tree_util

from jax.tree_util import PyTreeDef
from jax._src.tree_util import broadcast_prefix, prefix_errors, equality_errors, KeyPath
from typing import Tuple, Optional, Union, Any, List, Dict, Sequence
from .tensor_validator import TensorValidator
from .core import Validator
from dataclasses import dataclass





def find_compatible_tree_path(schema_path: KeyPath,
                               operand_path: KeyPath
                               ) -> KeyPath:
    """
    Compare two tree KeyPaths, and figure out what portions
    were similar.

    A keypath is a tuple of KeyEntries representing the tree path. It is
    a jax object, and capable of turning itself into a string.

    :param schema_path: The KeyPath for the schema. A keypath is a tuple of KeyEntries representing the tree path
    :param operand_path: The KeyPath for the operand.
    :return: A new keypath, consisting of only what is common
    """
    output = []
    for schema_node, operand_node in zip(schema_path, operand_path):
        if schema_node != operand_node:
            break
        output.append(schema_node)
    return tuple(output)

def format_keypath_message(keypath: KeyPath,
                            indent: str,
                            last_is_leaf: bool,
                            first_is_root: bool
                           ) -> str:
    """
    Formats a keypath into a string of indented entries with
    one branch or child for each line.

    :param keypath: The keypath to format.
    :param indent: What indent to put before each line
    :param last_is_leaf: Whether to mark the last entry in the keypath as a leaf
    :param first_is_root: Whether to mark the first entry as root
    :return: The formatted message
    """
    output = []
    for i, key in enumerate(keypath):
        if i == (len(keypath) - 1) and last_is_leaf:
            header = "leaf at: "
        elif i == 0 and first_is_root:
            header = "root at: "
        else:
            header = "branch at: "
        msg = header + str(key)
        output.append(msg)
    header = "\n" + indent
    message = header.join(output)
    message = message[1:]  # Strip out the first newline
    return message



def generate_exception(schema_tree,
                       operand_tree,
                       common_path: KeyPath,
                       ):


    compatible_path = find_compatible_tree_path(schema_path, operand_path)
    tree
class PyTreeShapeException(RuntimeError):
    def __init__(self,
                 details_message: str,
                 schema_path: KeyPath,
                 operand_path: KeyPath
                 ):

        message = """\
        An issue arose while validating the pytree shape
        
        {details_message}
        
        Keep in mind if you are using dictionaries, the order matters!
        
        Since the schema and operand did not match, the pytrees are 
        incompatible.
        """
        message = textwrap.dedent(message)
        message = message.format(details_message=details_message)

        self.message = message
        self.schema_path = schema_path
        self.operand_path = operand_path
        super().__init__(message)

class SuspectBranchOutOfOrderException(PyTreeShapeException):
    """
    Called when the schema is violated because:

    A leaf or collection of leaves was expected at a particular
    branch location in a particular order. However, while a compatible
    leaf was found, it was in the wrong order.
    """
    def __init__(self,
                 schema_leaf_path: KeyPath,
                 operand_leaf_path: KeyPath
                 ):
        """

        :param schema_leaf_path:
        :param operand_leaf_path:
        """

class SuspectBranchMissingException(PyTreeShapeException)
    """
    Called when schema is violated because:
    
    Branch pointed to s
    """

class MissingLeafException(PyTreeShapeException):
    """
    Called when the schema is violated because a
    leaf expected by the schema was not provided
    on the operand.
    """

    def __init__(self,
                 schema_path: KeyPath,
                 operand_path: KeyPath):

        indent = "   "
        compatible_tree_path = find_compatible_tree_path(schema_path, operand_path)
        incompatible_schema_path = schema_path[len(compatible_tree_path):]
        incompatible_operand_path = operand_path[len(compatible_tree_path):]

        compatible_path_message = format_keypath_message(compatible_tree_path, indent,
                                                         last_is_leaf=False, first_is_root=True)
        schema_path_difference_message = format_keypath_message(incompatible_schema_path, indent,
                                                                last_is_leaf=True, first_is_root=False)
        operand_path_difference_message = format_keypath_message(incompatible_operand_path, indent,
                                                                 last_is_leaf=True, first_is_root=False)

        details_message = """
        The schema and operand trees match correctly up to:
        {compatible_path_message} 

        Beyond this point, it was expected that there would be descendents on operand tree
        following this path to a leaf or broadcast location:
        {schema_path_difference_message}
           
        However, starting from the common branch, the operand next had an observed leaf along path:
        {operand_path_difference_message}
        """
        details_message = textwrap.dedent(details_message)
        details_message = details_message.format(
                                             compatible_path_message=compatible_path_message,
                                             schema_path_difference_message=schema_path_difference_message,
                                             operand_path_difference_message=operand_path_difference_message)
        super().__init__(details_message, schema_path, operand_path)


class ExcessLeafException(PyTreeShapeException):
    """
    Called when we make it all the way through all the
    leaf nodes specified by schema, but there are unhandled
    leaf nodes in operand.

    This means only one thing: The tree shape is incompatible, and we only
    caught it when we saw there was an extra leaf in the operand tree
    but not the schema
    """

    def __init__(self,
                 schema_path: KeyPath,
                 operand_path: KeyPath):

        compatible_path = find_compatible_tree_path(schema_path, operand_path)

        details_message = """
        The schema expected there to be no more children 
        on branch:
        {schema_branch}
        
        However, an additional
        """

        compatible_tree_path




class PyTreeValidator(Validator):


    def get_schema(self, **kwargs)->Any:
        if isinstance(self.schema, Schema):
            return self.schema.schema
        elif self.schema in kwargs:
            return kwargs[self.schema].schema
        else:
            raise RuntimeError(f"Schema of name '{self.schema}' was not provided among kwargs")
    def _validate(self, schema: Union[Validator, None], operand: Any):
        schema = self.a
        try:
            tree_util.tree_map()
    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        schema = self.get_schema(**kwargs)
        schema = broadcast_prefix(schema, operand, lambda x : x is None)


        schema_leaves, operand_leaves = self.get_leaves(schema, operand)


        while schema_leaves and operand_leaves:
            schema_path, schema_validator = schema_leaves.pop()
            operand_path, operand_leaves = operand.pop()


            while schema_leaves and operand_leaves:

                operand_path, operand_value = operand_leaves.pop()

                if self._is_same_path(schema_path, operand_path):
                    schema_path, schema_validator = schema_leaves.pop()
                    validator = self.apply_nonlocal_validation(validator)
                elif not self._is_broadcastable(schema_path, operand_path):
                    msg = "Schema tree was not broadcast with, or the same shape as, operand tree"
                    raise InternalTreeValidatorError(msg, None)
    def __init__(self, schema: Union[str, Schema]):
        self.schema = schema