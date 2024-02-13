import jax
import textwrap
from jax import tree_util
from jax.tree_util import PyTreeDef, KeyPath, KeyEntry
from typing import Tuple, Optional, Union, Any, List, Dict, Sequence
from .tensor_validator import TensorValidator, PassthroughValidator
from dataclasses import dataclass
@dataclass
class Schema:
    schema: Any

class InternalTreeValidatorError(Exception):


    def __init__(self,
                 msg: str,
                 suberror: Optional[Exception]
                 ):
        super().__init__(msg)
        self.suberror = suberror
        self.msg = msg
class TreeValidatorError(Exception):
    pass

class PreliminaryValidateLeaves(TensorValidator):
    def validate(self,
                 operand: Tuple[Sequence, Sequence],
                 **kwargs
                 ) -> Optional[Any]:

        return operand
class PyTreeValidator(TensorValidator):
    """


    fields
    ------

    schema: Either a string, indicating the kwarg to look for a schema in,
            or a Schema itself. Represents where to look for tree structure,
            and possibly tensor validators
    header: An optional tensor validator. The Header will be applied to ALL
            tensors found within a PyTree. It will be applied BEFORE any
            schema-specific values.
    tail: An optional tensor validator. The Tail will be applied to ALL
          tensors found within a PyTree. It will be applied AFTER any
          schema-specific validator.
    broadcast: A bool. Indicates whether or not to broadcast schemas

    manipulation
    ------------

    """

    def __init__(self,
                 schema: Union[str,Schema],
                 header: Optional[TensorValidator] = None,
                 tail: Optional[TensorValidator] = None,
                 broadcast_schema: bool = True,
                 ):

        """
        :param schema: The schema to draw from. Must either be an actual
                       Schema, locking you into that pattern, or a string
                       indicating the kwarg to look for it in
        :param header: Optional. A Header is a TensorValidator that will be
                       applied to ALL branch
        :param tail:
        :param broadcast_schema:
        """
        super().__init__()
        assert isinstance(schema, (str, Schema))
        self.schema = schema
        self.header = header
        self.tail = tail
        self.broadcast = broadcast_schema
    def apply_nonlocal_validation(self,
                        validator: Optional[TensorValidator]
                        )->TensorValidator:
        if validator is None:
            validator = PassthroughValidator()
        validator = self.header & validator & self.tail
        return validator

    def _is_same_path(self,
                      schema: Tuple[KeyEntry, ...],
                      operand: Tuple[KeyEntry, ...]
                      )->bool:
        return schema == operand
    def _is_broadcastable(self,
                          schema_path: Tuple[KeyEntry,...],
                          operand_path: Tuple[KeyEntry,...]
                          )->bool:
        # A schema is broadcastbale with an operand
        # when the path, up to the schema node, is
        # the same
        length = len(schema_path)
        return schema_path == operand_path[:length]
    def _format_path_message(self,
                          path: Tuple[KeyEntry,...],
                          )->str:
        path_message = "\n"
        for item in path:
            path_message += f"{item}\n"
        return path_message
    def get_schema(self, **kwargs)->Schema:
        """
        Gets the schema from either
        :param kwargs:
        :return:
        """
        if isinstance(self.schema, Schema):
            schema = self.schema
        elif self.schema not in kwargs:
            msg = f"""
            An issue occurred retrieving the schema for
            PyTreeValidator. 

            Operating in Kwarg mode,  A keyword argument of 
            {self.schema} was defined, but this was not in the
            provided kwargs
            """
            raise InternalTreeValidatorError(msg, None)
        elif not isinstance(kwargs[self.schema], Schema):
            msg = f"""
            An issue occurred retrieving the schema for
            PyTreeValidator
            
            Operating in kwarg mode, the keyword of 
            name '{self.schema}' was found, but was 
            found to be corrolated with a feature of type
            '{type(kwargs[self.schema])}, not Schema            
            """
            raise InternalTreeValidatorError(msg, None)
        else:
            schema = kwargs[self.schema]
        return schema
    def get_leaves(self,
                   schema: PyTreeDef,
                   operand: PyTreeDef
                   )->Tuple[
                             List[Tuple[KeyPath, Any]],
                             List[Tuple[KeyPath, Any]]
                            ]:
        # To get all the leaves we need to succesfully
        # compare the two trees, we MUST capture leafs
        # ending in None. We define a special utility function
        # then capture and return

        include_none_as_leaf = lambda x: x is None
        schema_leafs = tree_util.tree_leaves_with_path(schema, include_none_as_leaf)
        operand_leafs = tree_util.tree_leaves_with_path(operand)

        if len(schema_leafs) > len(operand_leafs):
           msg = f"""
           An issue was detected while retrieving
           the leaves.
           
           The schema under consideration was provided
           with a number of leaves, including nones,
           equal to '{len(schema_leafs)}. However, the
           operand had less leaves, at '{len(operand_leafs)}'
           
           This must mean incompatibility, as the schema
           can neither be broadcast nor exactly match.
           """
           msg = textwrap.dedent(msg)
           raise InternalTreeValidatorError(msg, None)

        return schema_leafs, operand_leafs

    ## Fufill contract
    #
    # When subclassing tensor_validator we promise
    # to implement validate, make_message, and make_exception
    def validate(self, operand: PyTreeDef, **kwargs)->Optional[Tuple[Dict[str, Any],
                                                                     Optional[Exception]]
                                                               ]:
        """

        :param operand:
        :param kwargs:
        :return:
        """


        # Set up certain variables as outer-level
        # locals. This will prove useful later on
        # if we throw an error. It also ensures
        # variables are shared across the try/except
        schema = None
        schema_path, schema_validator = None, None
        operand_path, operand_value = None, None

        try:
            # Try to validate. If all of the code runs
            # without throwing, validation was successful.
            #
            # The only kind of error that is thrown by code
            # in this section by the class itself is the
            # InternalTreeValidatorError which will contain
            # the primary issue.

            schema = self.get_schema(**kwargs)
            schema = schema.schema
            schema_leaves, operand_leaves = self.get_leaves(schema, operand)

            schema_path, schema_validator = schema_leaves.pop()
            validator = self.apply_nonlocal_validation(schema_validator)
            while schema_leaves and operand_leaves:

                operand_path, operand_value = operand_leaves.pop()

                #If not tree broadcasting, we update the path
                #
                # If we ARE tree broadcasting, the same schema validator
                # needs to be applied across many operand_values, so we
                # do not update the schema_path or validator.
                #
                # If neither, we have encountered a broken tree. Tell the
                # user.
                #
                # Keep in mind most of the error details are not tacked on
                # here. Information such as the schema path and such
                # are tacked on in the except branch and elsewhere

                if self._is_same_path(schema_path, operand_path):
                    schema_path, schema_validator = schema_leaves.pop()
                    validator = self.apply_nonlocal_validation(validator)
                elif not self._is_broadcastable(schema_path, operand_path):
                    msg = "Schema tree was not broadcast with, or the same shape as, operand tree"
                    raise InternalTreeValidatorError(msg, None)

                # Apply our actual validator. Raise if needed
                validation_result = validator(operand_value, **kwargs)
                if validation_result is not None:
                    msg = "Validation failed when applied to operand. See child error messages"
                    raise InternalTreeValidatorError(msg, None)

        except InternalTreeValidatorError as err:
            # All internal error emissions are performed in terms of thre
            # InternalTreeValidatorError, which will contain the issue and
            # any subissues as well.
            #
            # In this location, we use that information, along with the current
            # environment, to tack on a bunch more detaisl.

            info_dictionary = {"issue": err.msg}
            if schema_path is not None:
                info_dictionary["schema_path"] = schema_path
            if operand_path is not None:
                info_dictionary["operand_path"] = operand_path
            if operand_value is not None:
                info_dictionary["operand_type"] = type(operand_value)
            return info_dictionary, err.suberror

    def make_message(self,
                     operand: Any,
                     arguments: Tuple[Dict[str, Any], Optional[Exception]],
                     context_string: str,
                     **kwargs) -> str:
        message = "An issue occurred while validating a pytree with a PyTreeValidator \n"
        message_construction_info, _ = arguments
        if "schema_path" in message_construction_info:
            msg = f"""
            The issue occurred while examining a schema
            with node pointing to:
            """
            msg = textwrap.dedent(msg)
            path_msg = self._format_path_message(message_construction_info["schema_path"])
            path_msg = textwrap.indent(path_msg,"    ")
            msg += path_msg
            message += msg
        if "operand_path" in message_construction_info:
            msg = f"""
            The issue occurred while examine an operand
            branch with node pointed to by:
            """
            msg = textwrap.dedent(msg)
            path_msg = self._format_path_message(message_construction_info["operand_path"])
            path_msg = textwrap.indent(path_msg, "   ")
            msg += path_msg
            message += msg
        if "operand_type" in message_construction_info:
            msg = f"""
            The operand that failed was of type:
            """
            type_message = f"    {message_construction_info['operand_type']}"
            msg = textwrap.dedent(msg)
            msg = msg + type_message
        if "issue" in message_construction_info:
            msg = textwrap.dedent(message_construction_info["issue"])
            msg = textwrap.indent(msg, "    ")
            msg = "The actual issue was:\n" + msg
            message += msg
        return message
    def make_exception(self,
                       message: str,
                       arguments: Tuple[Dict[str, Any], Optional[Exception]]
                       ) -> Exception:
        """
        Fufill contract to be a tensor validator
        """
        _, sub_exception = arguments
        error = TreeValidatorError(message)
        if sub_exception is not None:
            error.__cause__ = sub_exception
        return error