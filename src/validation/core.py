import textwrap
import cachetools
import cachetools.keys

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Tuple, Callable, Type, Hashable, Generator, Union

from .state import get_success_callback, get_exception_callback, get_cache
from .types import Operand

import jax.tree_util
from jax import tree_util
from jax.tree_util import PyTreeDef

####
# Define validation error, and validation functions.
#
# Generally, these errors are used only if the user
# set something up wrong.
########




class ValidatorException(Exception):
    """
    Raised when something goes wrong during execution
    of a validator that is not connected to the origina
    validation target.

    An example would be a user misidentifying an operand
    """

    def __init__(self,
                 subclass: Type,
                 task: str,
                 message: str
                 ):
        output = f"A ValidatorException occurred while doing: '{task}' \n"
        output = output + textwrap.indent(message, "    ")

        super().__init__(output)
        self.subclass = subclass
        self.task = task
        self.message = output


def format_exception_message(
        subclass_name: str,
        subtask: str,
        details: str
) -> str:
    """
    Validator exceptions are pretty formatted to have a main issue

    :param subclass_name: The name of the subclass we encountered the issue on
    :param subtask: The subtask that we encountered an issue on, like checking probabilities
    :param details: The details on the issue
    :return: A formatted validator exception
    """


    # Merge the subtask, subclass, and details information together.
    # Get something that will look decent as the main body of the
    # error.
    header = f"An issue occurred while doing: '{subtask}'\n"
    body = textwrap.dedent(details)
    tail = f"This occurred on validator of type: '{subclass_name}'"
    message = header + "\n" + body + "\n" + tail

    # Create the error, and return it
    return message


def create_initialization_exception(subclass_name: str,
                                    subtask: str,
                                    details: str
                                    ) -> ValidatorException:
    """
    Creates an initialization exception, specifying the exception occurred during
    initialization of the class. Subtask, and details, meanwhile will end up
    merged into

    :param subclass_name: The name of the validator that is having issues
    :param subtask: The initialization task we are troubled by
    :param details: The details of that initialization task
    :return: The exception
    """
    task = "trying to initialize the class"
    message = format_exception_message(subclass_name, subtask, details)
    return ValidatorException(subclass_name, task, message)

class LinkListNode:
    """
    A statically defined linked list node mechanism,
    this is designed to implement the linked list
    part of the equation.

    """

    ##
    # Define the central features of each node
    #
    # There exists a next_node feature and a
    # rebind factory. This is particular to the node
    #
    # There also exists a attr class and
    # instance cache that will be common across all LinkListNode
    # instances
    ###
    next_node: 'LinkListNode'
    NodeBindingFactory: 'NodeBindingFactory'
    attr_cache = get_cache("LinkAttrCache")
    instance_cache = get_cache("LinkInstanceCache")

    ##
    # Define some relatively simple properties.
    ##

    @property
    def has_next_node(self)->bool:
        return self.next_node is not None

    ###
    # Setup various utility methods.
    #
    # This is very important as it ensures the caching mechanism
    # can distinguish between different methods
    ###
    def method_key_factory(self, name: str):
        """
        Creates a method key factory, which has the
        name of the method included in the hashing process

        This should hopefully ensure we do not confuse different
        methods with the same arguments
        :param name: The name of the method
        :return: A method key, bound to the method with that name
        """

        def method_key(self, *args, **kwargs):
            args = [self, name, *args]
            return cachetools.keys.hashkey(*args, **kwargs)

        return method_key

    def rebind_node(self, node: Optional['LinkListNode'])->'LinkListNode':
        """
        A very important method, this rebinds a provided node to point
        at a different node instead

        :param node: The node to point at
        :return: A node with the same value, pointing to the new location
        """
        return self.__factory_key.bind_key(node)


    ##
    # Setup the attributes and features that need to be cached
    # in the attr cache.
    #
    # These will not  return new instances
    ##
    @cachetools.cached(attr_cache, method_key_factory("length"))
    def __len__(self) -> int:
        if not self.has_next_node:
            return 1
        return len(self.next_node) + 1
    @cachetools.cached(attr_cache, method_key_factory("key"))
    def key(self)->Tuple['NodeBindingFactory', ...]:
        if len(self) == 1:
            return (self.__factory_key,)
        return tuple([self.__factory_key, *self.next_node.key()])
    @cachetools.cached(attr_cache, method_key_factory("hash"))
    def __hash__(self) -> int:
        return hash(self.key())

    ##
    # Other actions may result in the return of an instance or even
    # a new instance. These are cached in the instance cache instead
    ##
    def __iter__(self)->Generator['LinkListNode']:
        yield self
        if self.has_next_node:
            for item in iter(self):
                yield item

    @cachetools.cached(attr_cache, method_key_factory("as_list"))
    def as_keylist(self)->List['NodeBindingFactory']:
        return list(self.key())

    @cachetools.cached(instance_cache, method_key_factory("from_list"))
    @classmethod
    def from_keylist(cls, items: List['NodeBindingFactory'])-> 'LinkListNode':
        if len(items) == 1:
            subnode = None
        else:
            subnode = cls.from_keylist(items[1:])
        return items[0].bind_key(subnode)

    # TODO: The following is not terribly elegant
    #
    # It gets the job done, but might need replacing if cache rebuilds happen often.

    @cachetools.cached(instance_cache, method_key_factory("getitem"))
    def __getitem__(self, item: Union[int, slice])->'LinkListNode':
        return list(self.__iter__())[item]
    @cachetools.cached(instance_cache, method_key_factory("append"))
    def append(self, item: 'LinkListNode')->'LinkListNode':
        items = self.as_keylist()
        items.append(item)
        return self.from_keylist(items)
    @cachetools.cached(instance_cache, method_key_factory("insert"))
    def insert(self, index: int, value: 'LinkListNode')->'LinkListNode':
        items = self.as_keylist()
        inserting = value.as_keylist()
        items = items[:index] + inserting + items[index:]
        return self.from_keylist(items)

    def __init__(self,
                 next_node: 'LinkListNode',
                 node_key_factory: 'NodeBindingFactory',
                 ):
        self.next_node = next_node
        self.__factory_key = node_key_factory



##
# Define tree util node spec class
###

@dataclass(frozen=True)
class ValidatorNodeSpec:
    type: Type['Validator']
    args: Any
    kwargs: Any

class AbstractValidator(ABC):
    """
    Define the abstract protocol that will be fufilled
    by subclasses here
    """
    @abstractmethod
    def chain_predicate(self, operand: Operand, **kwargs: Any)->bool:
        """
        A logical method that a subclass of validator must fufill in order
        for its logic to operate properly

        The chain predicate controls whether or not to continue validation if
        the validation mechanism is passed.

        :param operand: The operand, in case chain predicate depends on it
        :param kwargs: The current kwargs. Can be used during condition
        :return: A bool or scalar bool array
        """
        pass

    @abstractmethod
    def predicate(self, operand: Any, **kwargs) -> bool:
        """
        Abstract method a subclass must implement. This
        should return true if validation is passed, and
        false otherwise

        :param operand: The operand that failed
        :param kwargs: The kwargs situation for the operand
        :return: A bool or scalar bool array
        """
        pass

    @abstractmethod
    def create_exception(self, operand: Any, **kwargs) -> Exception:
        """
        Abstract method that subclasses must implement to perform validation checks
        on the given operand. This method should return an exception to be raised and otherwise handled. It
        will only be triggered when the predicate fails.

        It should NOT raise the exception itself.

        :param operand: The operand to be validated.
        :param kwargs: Additional keyword arguments that may be needed for validation.
                       These arguments are accessible to all validators in the chain.
        :return: A bool or scalar bool array, and an exception to raise if failing
        """
        pass

    @abstractmethod
    def handle_exception(self, exception: Exception, **kwargs) -> Exception:
        """
        Handles an exception detected from further down in the
        validation chain. Does something with it. Must pass back
        an exception, but the exception need not be the same one.

        :param exception: The exception from downstream. You may log, throw,
                          or otherwise cause side effects. It is recommended
                          to log into kwargs if you do so
        :return: Either an error, in which case the error chain continues, or None,
                 which suppresses the error. You may even transform the original
                 error, and return the new error.
        """
        return exception



#####
# It is difficult to contain all mechanisms of core in a single class without it getting unwieldy
#
# We define the following node mechanism to allow us to wrap the user-provided functions
####

class SubclassMethodsInterface:
    """
    This class is a value validation-focused node that contains
    the features of relevance extracted from a particular subclass.

    For the most part, this just means the bound methods responsible
    for performing validation. However, it also may include in the
    future other methods, and at the moment includes the name of
    the subclass so error messages will look pretty.

    Regardless, anything that interacts with a subclass's code
    or fields must go through this class.

    Methods
    =======

    Ignoring internal methods, the following are provided

    - predicate: Whether validation passed
    - create_exception: A bound exception factory function
    - handle_exception: Two things, first emit side effects and also emit new (or old) error
    - chain_predicate: Whether to continue validation/.

    Initialization
    ==============

    Initialization consists of providing the subclass name, and then
    the subclass's version of the validation function listed above.

    """
    _exception_predicate_name: str = 'predicate'
    _exception_factory_name: str = 'create_exception'
    _exception_handle_name: str = 'handle_exception'
    _exception_chain_name: str = 'chain_predicate'
    _node_cache: Dict[Hashable, 'SubclassMethodsInterface'] = {}

    #####
    # Create error formatting functions. We will have many
    # cases where we need to either tell the user you gave us code
    # that does not run, or returned the wrong type.
    #
    # Following DRY, these are extracted into common methods
    ####
    def create_subclass_code_exception(self,
                                       code_feature: str,
                                       details: str) -> ValidatorException:
        """
        A helper class, it tries
        :param subclass_name: The validator subclass of concern
        :param code_feature: Which code feature, such as 'predicate', is misbehaving
        :param details: Any error message details
        :return: The exception
        """
        task = "Validating the user provided operand"
        subtask = f"Executing or using subclass provided code called '{code_feature}'"
        message = format_exception_message(self.subclass_name, subtask, details)
        return ValidatorException(self.subclass_name, task, message)

    def create_subclass_code_did_not_run_exception(self,
                                                   code_feature: str
                                                   ) -> ValidatorException:
        details = f"""\
        The provided code did not run successfully. It is possible that
        your code was malformed, you did not provide needed kwargs, or even
        that jit is acting up
        """
        return self.create_subclass_code_exception(code_feature, details)

    def create_subclass_code_returned_wrong_type_exception(self,
                                                           code_feature: str,
                                                           required: str,
                                                           observed: Any,
                                                           ) -> ValidatorException:
        details = f"""\
        The provided code ran just fine, however it did not return the 
        correct type. The required type was '{required}', but what was
        observed was '{type(observed)}
        """
        return self.create_subclass_code_exception(code_feature, details)
    #####
    # Validation is required for every one of the important responsibilities
    # subclasses
    #
    # These provide access points that are validated.
    ####

    def predicate(self, operand: Operand, **kwargs: Any)->bool:
        """
        Checks if the predicate is satisfied. Also, provides
        feedback when the user-provided predicate is insane

        :param operand: The operand to validate
        :param kwargs: The kwargs conditioning the validation
        :return: A bool or bool array
        :raise ValidatorException: If the predicate specified did not run
        :raise ValidatorException: If the predicate ran, but did not return a bool type.
        """
        try:
            outcome = self._predicate(operand, **kwargs)
        except Exception as err:
            raise self.create_subclass_code_did_not_run_exception(self._exception_predicate_name) from err
        if not isinstance(outcome, bool):
            raise self.create_subclass_code_returned_wrong_type_exception(self._exception_predicate_name,
                                                                          required='bool',
                                                                          observed=outcome)
        return outcome

    def create_exception(self, operand: Operand, **kwargs: Any)->Exception:
        """
        Create an exception which will be managed by the chain of
        responsibility, and validate that the exception is sane

        :param operand: The operand to validate
        :param kwargs: The Kwargs conditioning the validation
        :return: An exception for the situation
        :raise ValidatorException: If the user-provided create exception mechanism did not run
        :raise ValidatorException: If the the user provided mechanism ran, but did not return 
                an exception
        """
        try:
            outcome = self._create_exception(operand, **kwargs)
        except Exception as err:
            raise self.create_subclass_code_did_not_run_exception(self._exception_factory_name) from err
        if not isinstance(outcome, Exception):
            raise self.create_subclass_code_returned_wrong_type_exception(self._exception_factory_name,
                                                                          required='Exception',
                                                                          observed=outcome)
        return outcome


    def handle_exception(self, exception: Exception, **kwargs: Any)->Exception:
        """
        Handles some action that an exception can cause or encourage

        :param exception: The exception to handle
        :param kwargs: The kwargs influencing this exception
        :return: An exception, either the same one or a different one
        :raise ValidatorException: If the user-provided handle exception mechanism could not be run
        :raise ValidatorException: If the the user provided mechanism ran, but did not return
                an exception
        """
        try:
            outcome = self._handle_exception(exception, **kwargs)
        except Exception as err:
            raise self.create_subclass_code_did_not_run_exception(self._exception_handle_name) from err
        if not isinstance(outcome, Exception):
            raise self.create_subclass_code_returned_wrong_type_exception(self._exception_handle_name,
                                                                          required="Exception",
                                                                          observed = outcome)
        return outcome

    def chain_predicate(self, operand: Exception, **kwargs: Any)->bool:
        """
        Checks into whether the chain predicate says to continue further in the
        chain of responsibility

        :param operand: The operand being validated
        :param kwargs: The kwargs conditioning the validation
        :return: A bool or bool array
        :raise ValidatorException: If the user-provided chain predicate mechanism did not run
        :raise ValidatorException: If the the user provided mechanism ran, but did not return
                a bool
        """
        try:
            outcome = self._chain_predicate(operand, **kwargs)
        except Exception as err:
            raise self.create_subclass_code_did_not_run_exception(self._exception_chain_name) from err
        if not isinstance(outcome, bool):
            raise self.create_subclass_code_returned_wrong_type_exception(self._exception_chain_name,
                                                                          required='bool',
                                                                          observed=outcome)
        return outcome

    def __init__(self, subclass: AbstractValidator):

        self.subclass_name = type(subclass).__name__
        self._predicate = subclass.predicate
        self._create_exception = subclass.create_exception
        self._handle_exception = subclass.handle_exception
        self._chain_predicate = subclass.chain_predicate

@dataclass(frozen=True)
class NodeBindingFactory:
    """
    A validator node may be subclassed, and the subclass must
    be reproducable multiple times in factory-like fashion
    but possibly pointing towards a different node.

    """
    ##
    # Define the fields that the constructor key can store
    # These will be some extra info for pretty printing, and
    # the features needed to rebuild the node, sans what the
    # next link would be
    ##

    name: str
    cls_type: Type['ListNode']
    params_flat: Tuple[Hashable, ...]
    params_treedef: PyTreeDef
    def __hash__(self)->int:
        key = (self.cls_type, self.params_flat, self.params_treedef)
        return hash(key)

    ###
    # Define the methods usable for making a key,
    # or binding the key back into a node
    ##
    @classmethod
    def make_key(cls, node: 'ListNode', *args, **kwargs)-> 'ConstructorNodeKey':
        """
        Creates a constructor key capable of rebuilding the current
        node if needed, and capable of representing the current
        node in hashable situations.

        :param instance: The instance to represent
        :param args: The args used to create the instance
        :param kwargs: The kwargs used to create the instance
        :return: A constructor key, related to the situation
        """
        cls_type = node.__class__
        name = cls.__name__
        constructor_params = (args, kwargs)
        flat_constructor_params, constructor_treedef = jax.tree_util.tree_flatten(constructor_params)
        return cls(name=name,
                   cls_type=cls_type,
                   params_flat = tuple(flat_constructor_params),
                   params_treedef=constructor_treedef)
    def bind_key(self, node: Optional['ListNode'])->'ListNode':
        """
        Takes an existing constructor key and binds it to a particular node,
        returning an instance.

        :param node: The next node to consider. Can be a listnode or none
        :return: The new listnode, with same arguments and bound to Node
        """
        args, kwargs = jax.tree_util.tree_unflatten(self.params_treedef, self.params_flat)
        instance = self.cls_type(node, *args, **kwargs)
        return instance


HandleCallback = Callable[[Exception, ...], Exception]
SuccessCallback = Callable[[Operand, ...], None]
class LogicNode:
    """
    Background
    ==========

    Internally and conceptually, validation is implemented
    using the chain-of-responsibility pattern. This means there is a
    linked list, and each element of the linked list has a particular
    responsibility

    This implements the linking mechanism, and also implements the
    logic to bind the list of links into a

    Purpose
    =======

    This class is that linked list.

    It is dedicated to providing logic for the validator-specific
    mechanisms, and for calling into the next node if it exists. It
    must also call into the success or failure callbacks, if provided.

    Logic - Recurrent vs actual
    ===========================

    Jax is a little tricky to run flow control against if you
    want to maintain jit compatibility.

    To ensure compatibility is maintained, we use flow control with
    a switch based architecture, and have switch cases validation completely
    passed, validation failed

    Additionally, exception generation and handling is executed under python
    by means of a debug callback.
    """



    ####
    # It is exceptionally difficult to logically organize
    # complex flow control using jax.
    #
    # We use a switch statement for the flow control cases
    # in this class. The following bits of code will
    # make the switch cases
    ####
    def did_validation_pass(self, operand: Operand, kwargs: Dict[str, Any])->bool:
        return self.subclass_functions.predicate(operand, **kwargs)
    def can_continue_chain(self, operand: Operand, kwargs: Dict[str,Any])->bool:
        if self.next_node is None:
            return False
        return self.subclass_functions.chain_predicate(operand, **kwargs)
    def make_switch_case(self, operand: Operand, kwargs: Dict[str, Any])->int:
        """
        Creates one of three switch cases in index format, since
        jax likes indices rather than branches. We define three cases -

        0: Failure case. An error was reached
        1: Success case. Validation is done
        2: Chain can be continued like normal
        """
        case = 0
        case = case + self.did_validation_pass(operand, kwargs) # Is one if passed, zero otherwise
        case = case + case*self.can_continue_chain(operand, kwargs) # In case prior passed,
        return case

    ####
    # Because jax is functional, making the logic work requires
    # a few factory methods and an exception branch must be
    # defined here
    ####

    def wrap_handle_callback(self, handle_callback: HandleCallback)->HandleCallback:
        """
        Wraps a handle callback to instead first call into
        this node's handle method, then call into the last
        node.

        :param handle_callback: The current handle callback
        :return: The new wrapped version
        """
        def wrapped_handle_callback(exception: Exception, **kwargs):
            # First runs the handle method on this node, then
            # calls into the handle callback from the last node
            exception = self.subclass_functions.handle_exception(exception, **kwargs)
            handle_callback(exception, **kwargs)
        return wrapped_handle_callback

    def run_exception_branch(self,
                             operand: Operand,
                             handle_callback: HandleCallback,
                             success_callback: SuccessCallback,
                             kwargs: Dict[str, Any]
                             ):
        """
        This is part interface, part logic, and is executed if an exception
        state is reached. It will create the exception, then call into the
        handling callback
        """
        exception = self.subclass_functions.create_exception(operand, **kwargs)
        handle_callback(exception, **kwargs)

    ###
    # Primary validation logic
    ###
    def run_validation(self,
                       operand: Operand,
                       handle_callback: HandleCallback,
                       success_callback: SuccessCallback,
                       kwargs: Dict[str, Any]) -> object:
        ##
        # Validation is performed with what is effectively a recurrent
        # architecture, that calls into subsequent code. For reasons
        # of jax compatibility, a switch statement is used for flow control.
        # There are two base cases, then a recurrent case:
        #
        # switch_index: 0: Base case, exception detected
        # switch_index: 1: Base case, all validation passed
        # switch_index: 2: Recurrent case, calls into next node.
        #
        ###

        handle_callback = self.wrap_handle_callback(handle_callback)
        switch_index = self.make_switch_case(operand, kwargs)
        switch_cases = [jax.tree_util.Partial(jax.debug.callback, self.run_exception_branch),
                        success_callback,
                        self.next_node.run_validation]
        jax.lax.switch(switch_index,
                       switch_cases,
                       operand,
                       handle_callback,
                       success_callback,
                       kwargs
                       )

    def __init__(self,
                 subclass_bindings: SubclassMethodsInterface,
                 next_node: Optional['LogicNode'] = None
                 ):
        """
        This should rarely ever be called directly
        :param subclass_bindings: The subclass methods interface containing the subclass methods
        :param next_node: The next logic node in the linked list.
        """
        self.subclass_functions = subclass_bindings
        self.next_node = next_node
    def __call__(self,
                 operand: Operand,
                 final_exception_callback: Optional[HandleCallback],
                 final_success_callback: Optional[HandleCallback],
                 kwargs: Dict[str, Any],
                 ):
        """
        Executes the validation contained within the chain.

        :param operand: The operand to be validated
        :param final_exception_callback: The final callback to be called into when traversing the exception branch
        :param final_success_callback: The final callback to be called into when successful at validating
        :param kwargs: The kwargs, in dictionary form, that were captured when the user called
        """
        self.run_validation(operand, final_exception_callback, final_success_callback, kwargs)


class ListNode(AbstractValidator, LogicNode):
    """

    """

    attributes_cache = get_cache("ListNodeAttr")
    instance_cache = get_cache("ListNodeInstance")
    next_node: 'ListNode'




    ###
    # Define some important cache manipulation mechanism
    #
    # Primarily, we define the method key factory. This will insert
    # data on the name of a method into the cache hashing mechanism, ensuring
    # different methods called with the same arguments cannot be confused
    #
    # We also define here
    ##

    def __method_key_factory(self, name: str):
        """
        Creates a method key factory, which has the
        name of the method included in the hashing process

        This should hopefully ensure we do not confuse different
        methods with the same arguments
        :param name: The name of the method
        :return: A method key, bound to the method with that name
        """

        def method_key(self, *args, **kwargs):
            args = [self, name, *args]
            return cachetools.keys.hashkey(*args, **kwargs)

        return method_key

    ##
    # Define a few magic methods used to manipulate lists
    #
    # Since these traverse the list, they are cached in the attributes
    # cache mechanism.
    ##
    @cachetools.cached(attributes_cache, __method_key_factory("hash"))
    def __hash__(self) -> int:
        if not self.has_next_node:
            return 1
        return self.next_node.__detect_length() + 1
    @cachetools.cached(attributes_cache, __method_key_factory("length"))
    def __len__(self) -> int:
        if not self.has_next_node:
            return 1
        return self.next_node.__detect_length() + 1
    @cachetools.cached(attributes_cache, __method_key_factory("iter"))
    def __iter__(self):
        yield self
        if self.has_next_node:
            for item in self.next_node:
                yield item




    ###
    # Define internal linked list walking methods
    # used to compute various quantities and perform
    # various tasks.
    #
    # We also define the pytree conversions here as well
    ###
    @cachetools.cached()
    def chain_key(self)->Tuple[ConstructorNodeKey, ...]:
        """
        Creates a hashable key that completely represents the chain of events that
        lead to this node being in place. That will include the constructor key used
        by myself, and by child nodes.
        :return:
        """
        node_key = ConstructorNodeKey.make_key(self, self.__args, self.__kwargs)
        if not self.has_next_node:
            return (node_key,)
        return tuple([node_key, *self.next_node.chain_key()])


    @staticmethod
    def from_chain_key(chain_key: Tuple[ConstructorNodeKey, ...])->'ListNode':
        chain = list(chain_key)
        node = None
        while chain:
            key = chain.pop()
            node = key.bind_key(node)
        return node

    def tree_flatten_with_keys(self):
        raise NotImplementedError()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        raise NotImplementedError()

    ##
    # Define the node manipulation methods
    #
    # These tend to be very expensive, as they may
    # involve rebuilding the entire linked list,
    # and so are cached.
    ##
    @cachetools.cached(instance_cache, __method_key_factory("rebind_node"))
    def rebind_node(self, node: 'ListNode')-> 'ListNode':
        """
        Reproduces the values of the current node, but
        with the link pointing to a different next node

        :param node: The Node to point to
        :return: A new node pointing to that link instead
        """
        # Essentially, this is a factory mechanism that reproduces the
        # args and kwargs passed in, but with a different node
        return self.__class__(node,
                              *self.__args,
                              **self.__kwargs)
    @staticmethod
    def _append(self: 'ListNode', other: 'ListNode'):
        self_list = self.

    @cachetools.cached(instance_cache, __method_key_factory("append"))
    def append(self, node: 'ListNode')->'ListNode':
        """
        Appends a validation sequence onto another
        validation sequence

        :param node: The node to append
        :return: The revised sequence
        """
        self_list = self.chain_key
        if self.chain_representation is None:
        if not node.has_next_node:
            # Base case reached, rebind
            return self.rebind_node(node)
        # Append to subnode, then rebind.
        subnode = self.next_node.append(node)
        return self.rebind_node(subnode)

    @cachetools.cached(instance_cache, __method_key_factory("insert"))
    def insert(self, index: int, node: 'ListNode')->'ListNode':
        """
        Inserts one list into another without breaking
        the links.

        Returns a new instance
        :param index: The index to set to
        :param node: The node to insert there
        :return: The new list node
        """
        if index == 0:
            # Base case, append myself onto inserted chain
            return node.append(self)
        # Recurrent descent case. Travel to link, insert
        # there, rebuild with new case
        subnode = self.next_node.insert(index - 1, node)
        return self.rebind_node(subnode)

    def __init__(self,
                 next_node: 'ListNode',
                 *args,
                 **kwargs
                 ):

        # Define and initialize the logic of the LogicNode
        #
        # Also sets up the linked list for the first time
        subclass_bindings = SubclassMethodsInterface(self)
        constructor_node = ConstructorNodeKey(self.__class_)
        super().__init__(subclass_bindings, next_node)

        # Store away arguments and keyword arguments
        # These can be used in factory methods to create
        # new instances pointing to different nodes
        self.__args = args
        self.__kwargs = kwargs

        # Store away something else

    # Define properties and related phenomenon properties
    @property
    def has_next_node(self) -> bool:
        return self.next_node is not None

class ValdationExecuter:
    """
    This contains the validator linked list
    and callbacks. It is the actual interface
    used to call into the validation logic.
    """
    cache = get_cache("LogicNodesCache")
    @classmethod
    def build_list(cls, nodes: List[SubclassMethodsInterface]):
        """
        Builds the validator nodes, using the cache
        to shortcut the process wherever possible.

        :param nodes: The nodes list to build
        :return: The validator linked list logic.
        """

        if len(nodes) == 1:
            # If the passed node has been observed before, the cache
            # should just fetch it.
            #
            # this is the base case
            return cls.cache.fetch(LogicNode, nodes[0], None)

        # Fetch the subnode, and then build the current node.
        subnode = cls.cache.fetch(cls.build_list, nodes[1:])
        return cls.cache.fetch(LogicNode, subnode)

    def __init__(self,
                nodes: List['Validator'],
                ):
        """
        Initialization should consist of the nodes list,
        which is a list of SubclassFeaturesNodes, and
        the existing
        :param nodes: The list of nodes to execute during validation
        :param final_exception_callback: The final callback to execute when failing
        :param final_success_callback: The final callback to execute when success.
        """

        # Get context elements from global state.
        final_exception_callback = get_exception_callback()
        final_success_callback = get_success_callback()

        # Create default callbacks if required
        if final_exception_callback is None:
            final_exception_callback = lambda exception, **kwargs : None
        if final_success_callback is None:
            final_success_callback = lambda operand, **kwargs : None

        # Convert validators into subclass nodes

        nodes = [SubclassMethodsInterface(item) for item in nodes]

        # Store
        self.validator_function = self.build_list(nodes)
        self.final_exception_callback = final_exception_callback
        self.final_success_callback = final_success_callback
    def __call__(self,
                 operand: Operand,
                 **kwargs
                 ):
        """
        Executes validation.

        :param operand: The operand to validate
        :param kwargs: The kwargs conditioning the validation.
        """
        self.validator_function(operand,
                        self.final_exception_callback,
                        self.final_success_callback,
                        kwargs)

class Validator(AbstractValidator):

    ###
    # Two important methods are lazily initialized
    # if needed
    ###
    cache = get_cache("Validator")
    @property
    def validate(self):
        if self._validate is None:
            self.__post_init_hook__()
        return self._validate

    @property
    def links(self)->List['Validator']:
        if self._links is None:
            self.__post_init_hook__()
        return self._links

    def __post_init_hook__(self, nodes: Optional['Validator'] = None):
        if nodes is None:
            nodes = [self]
        self._links = nodes
        self._validate = ValdationExecuter(nodes)

    def rebind_links(self,
               nodes: List['Validator']
               )->'Validator':
        instance = self.cache.fetch(self.constructor, *self.__args, **self.__kwargs)
        instance.__post_init_hook__(nodes)
        return instance

    def append(self, validator: 'Validator')->'Validator':
        links = self.links + validator.links
        instance = self.rebind(links)
        return instance
    @classmethod
    def constructor(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        instance.__args = args
        instance.__kwargs = kwargs
        return instance
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        instance = cls.cache.fetch(cls.constructor, *args, **kwargs)

        instance.__args = args
        return instance

    def __init__(self, next_validator: Optional['Validator']):
    def __init__(self):
        self.__link_setup_run: bool = False
        self._links: List['Validator'] = [self]
        self._validate: Optional[ValdationExecuter] = None

    def append(self, validator: 'Validator')->'Validator':
        links = self._links
        links = links + validator.links
        instance = type(self)(*self.__args, **self.__kwargs)



# Begin main definition
class Validator(ABC):
    """
    Abstract base class for creating validators. Validators can be used to perform
    validation checks on operands and can be chained together to form complex validation
    logic. Each subclass must implement the `validate` method to define its validation behavior.

    The Chain of Responsibility
    ===========================

    Lets talk here about validator precedence, and the validation chain.
    Conceptually, Validators exist in a linked list, with it being the case that
    new validators are appended to the front of the list. This follows the chain
    of responsibility design patterns. For instance, you might have a validator called
    Throw, connected to one called IsFloatingDtype, connected to one called IsProbabilities. That
    would look something like:

    Throw->IsFloatingDtype->IsProbabilities

    Each validator's own responsibility is to

    a): Define .validate to specify what to check for in terms of the incoming operand, and return
        an Exception through .validate if failing, or otherwise None
    b): Define .handle to specify what to do with an Exception raised. Most of the time,
        and by default, the exception is just passed back. This sends it further back up
        the chain.
    c): Define chain_predicate, to specify based on mutable kwargs whether to continue
        checking further elements in the chain. By default, this is true, always indicating to continue.
        However, kwargs can modify the behavior if desired.

    It is also responsible for calling into it's children's validate to continue the chain, though
    this is not something the implementer of a validation node needs to worry about.

    When a validation action is requested, the internal logic first calls validate, and checks
    if an error occurs. If none did, it calls the next node in the list, which validates and calls ITS next
    node, and so on.

    Eventually, one of two things will happen. We will hit the end of the list, and see no error. In that
    case, a None is sent back up the stack, and nothing happens. Or, a validation condition will fail. In THIS
    case, the generated exception is returned, and the handle method will come into play.

    On the parent node of the list, the .handle method is called with the exception returned by the
    child, and the result of .handle is then returned as well. .handle can be used for things
    like suppressing the error (By returning None through handle), modify it
    (Make a new exception to return within handle) or even throwing or logging it. However,
    the majority of the time it makes sense to just have it return the error unchanged.

    Defining and using a validator
    ====================

    The definition phase involves defining what we actually want the
    validator to do.

    To do this, we must define abstract method
    'validate', and return either none, when validation passes,
    or an exception, if it fails.

    We may also setup the 'handle' function to let us know what
    to do when another validator downstream finds an error. Passing
    an error back will ensure it continues to be passed upstream,
    while not passing back will stop the error chain on that node.

    Note that regardless on whether you plan to use them or not,
    you still need to specify kwargs.

    One additional important detail. If you are passing in something
    as an __init__ function, it must be a pyree, that can be flattened
    by jax.tree_util.flatten, and that has hashable leaves.

    Defining a Simple example
    -------------------------

    A simple example which does not use __init__ is listed as follows

    ```
    class PositiveValidator(Validator):
        # This simple class works without an init function
        def validate(self, operand, **kwargs):
            if jnp.all(operand > 0):
                return None
            return Exception("Operand must be positive")
    ```

    Defining a validator using init
    ---------------------------------

    Setting up an __init__ call is optional, and may not be needed
    for simple validators. However, some validators, that should be setup once
    and used many times, may benefit from using an init. Lets see an example

    ```
    class IsDtype(Validator):
        #This sets up a threshold on the fly
        def __init__(self, dtype: jnp.dtype):
            self.dtype = dtype
        def validate(self, operand, **kwargs):
            if operand.dtype != self.dtype
                raise TypeError("Wrong Dtype")
    ```
    Defining a validator using runtime mutables
    ---------------------------------------------

    Sometimes, you may have features for which you do
    not details of until you actually run the program.
    Batch shape might be a good example.

    Such cases are also covered. You can configure your validate
    function to retrieve something out of kwargs, or accept a kwarg from
    upstream. Note that *args are not allowed.

    For example:

    ```
    class EnsureBatchShape(Validator):
        def validate(self,
                    operand: jnp.ndarray,
                    batch_shape: int,
                    **kwargs)->Optional[Exception]:
            if operand.shape[0] != batch_shape

    Defining how errors are dealt with using handle
    -----------------------------------------------

    Sometimes, you might want to do something
    with the error that has been detected, such as
    throw it or log it. This is what the function
    "handle" is for.

    Passing an error back out of handle will ensure it continues to be passed upstream,
    while not passing back will stop the error chain on that node. Lets see a few examples
    of how this could be used.

    ```
    class Log(Validator):
        # Log the event at this location by
        # using a callback
        def validate(self, operand, **kwargs)->Optional[Exception]:
            # Passthrough on the forward pass
            return None
        def handle(self, exception: Exception, logger)->Exception:
            logger.log(exception)
            return exception
    ```

    ```
    class Throw(Validator):
        # Just throw when I see an error

        def validate(self, operand, **kwargs)->Optional[Exception]:
            # Passthrough on the forward pass
            return None
        def handle(self, exception: Exception):
            throw exception
    ```
    Definign to Suppress_Errors
    ---------------------------
    One last thing worth mentioning is the utility of
    check_next. This is called before continuing
    on to the next node in the chain, and can be used
    to prevent any further computation in the chain from occuring at all

    ```
    class SuppressErrors(Validator):
        # Conditionally suppresses errors from further down the chain
        # if the suppress_errors kwarg bool was true
        def validate(self, operand, **kwargs)->Optional[Exception]:
            # Passthrough on the forward pass
            return None
        def chain_predicate(self,
                       suppress_errors: bool,
                       **kwargs):
            if None
    ```

    Validator Composition
    ======================

    Validators are designed to be composable, allowing multiple validators to be combined
    using logical operations. This composability enables the construction of complex validation
    logic by chaining simpler validators.

    Example usage:
        class PositiveValidator(Validator):
            def validate(self, operand, **kwargs):
                if operand > 0:
                    return None
                return Exception("Operand must be positive")

        class EvenValidator(Validator):
            def validate(self, operand, **kwargs):
                if operand % 2 == 0:
                    return None
                return Exception("Operand must be even")

        # Compose validators to check if an operand is both positive and even
        validator = PositiveValidator() & EvenValidator()

        # Use the composed validator
        error = validator(5)  # This will pass the PositiveValidator but fail the EvenValidator
        if error:
            print(error)

    """

    # These properties are created once, at time of
    # initialization, and are then never modified.
    #
    # Instead, new objects end up being returned.

    next_validator: 'Validator'
    hash_value: int
    __args: List[Any]
    __kwargs: Dict[str, Any]

    __exception_callback: Callable = lambda exception, **kwargs: None
    __success_callback: Callable = lambda operand: operand

    @property
    def has_next(self) -> bool:
        return self.next_validator is not None

    @classmethod
    def set_global_exception_callback(cls,
                                      callback: Callable[[Exception, ...], None]):
        """
        Sets a globally active exception callback connected to
        every validator.

        This callback will be provided
        :param callback: A callable that accepts an exception, and any number
                         of kwargs.
        """
        cls.__exception_callback = callback

    def get_root_exception_callback(self) -> Callable[[Exception, ...], None]:
        """
        Gets the root, or final, exception callback. If
        None, we make a lambda to accept the call
        :return:
        """
        if self.__success_callback is not None:
            return self.__success_callback
        return lambda exceptions, **kwargs: None

    @classmethod
    def set_global_success_callback(cls,
                                    callback: Optional[Callable[[Any, ...], None]]):
        """
        Sets up a globally active exception callback for every validator
        in existance. Conceptually, this goes off anytime no validaton fails

        :param callback: The callback, which should accept an operand and any
               number of kwargs
        """
        cls.__success_callback = callback

    def get_success_callback(self) -> Callable[[Any, ...], None]:
        """
        Gets the success callback, and returns a no-op lambda
        if not yet set

        :return: The success callback, which will accept an operand
        and any number of kwargs
        """
        if self.__success_callback is not None:
            return self.__success_callback
        return lambda operand, **kwargs: None

    #################
    # Define initialization routines.
    #
    # A large portion of the class complexity is the initialization routine. It is responsible
    # for setting up the linked lists.
    #
    # The parent class relies on a mixture of __new__ and __init_subclass__ for it's
    # own setup of the linked lists, and leaves __init__ alone for any subclass
    # implementing the contract
    #
    # It also creates a hash from the initialization properties and caches
    # the resulting instance, then will use that instance instead in future
    # runs
    #
    # We have to ensure any new subclasses are registered with jit
    #
    # There will also be planned logic involving a global head as well.
    #
    ###################
    def _get_constructor_parameters(self) -> Tuple[List[Any], Dict[str, Any]]:
        return self.__args, self.__kwargs

    @classmethod
    def _get_unique_class_identifier(cls) -> str:
        return f'{cls.__module__}.{cls.__name__}'

    @classmethod
    def _create_hash(cls,
                     constructor_treedef: PyTreeDef,
                     constructor_leaves: List[Any],
                     next_validator: Optional['Validator']
                     ):
        """
        Two validation linked lists will behave entirely equivalently
        if it is the case that:

        1): The nodes of the lists are, in order, the same class
        2): The nodes of the two lists are, in order, passed exactly the same arguments in init

        We define a hash function to create a unique hash based on
        the class of each node, the arguments that it was passed, and the arguments of the
        prior node. By this mechanism, we develop a function that will uniquely identify
        whether two validator linked lists are compatible or equivalent.
        """
        try:
            constructor_pytree_def_id = hash(constructor_treedef)
            constructor_arguments_id = hash(tuple(constructor_leaves))
            qualified_class_id = hash(cls._get_unique_class_identifier())
            node_linkage_id = hash(next_validator) if next_validator is not None else None
        except TypeError as err:
            subtask = "trying to hash leaves, treedef, and node"
            details = """\
            It is highly likely you provided a leaf which is not hashable as 
            a constructor argument. This is not allowed. 
            """
            raise create_initialization_exception(cls, subtask, details) from err
        representation = (constructor_pytree_def_id, constructor_arguments_id, qualified_class_id, node_linkage_id)
        return hash(representation)

    def __hash__(self) -> int:
        return self.hash_value

    def __init_subclass__(cls, **kwargs):
        # A small but important method that sets up each
        # subclass as it is brought online
        #
        # We must patch init anytime we subclass it to transparently accept and remove
        # "_next_validator" from the parameters that will be reaching the
        # user's __function__. The value '_next_validator' is sometimes passed
        # along when building a class by methods within this parent related to maintaining
        # the linked list. It is consumed in __new__. However, user __init__ functions will
        #  never need to know about that, and so we remove it before they see it.
        #
        # We also register the subclass with tree util.

        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            if "_next_validator" in kwargs:
                kwargs.pop("_next_validator")
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        jax.tree_util.register_pytree_node_class(cls)

    def __new__(cls, *args, _next_validator: Optional['Validator'] = None, **kwargs, ):

        # Process the incoming arguments. This means stripping
        # kwargs of _next_validator, which is not something the implementer
        # should see, quantify the tree structure of the incoming constructor
        # arguments so we can setup a factory, and making a cache id by hashing
        # the above.

        constructor_parameter_leaves, constructor_treedef = tree_util.tree_flatten((args, kwargs))
        next_validator: Optional[Validator]
        cache_id = cls._create_hash(constructor_treedef, constructor_parameter_leaves, _next_validator)

        # We either get an already existing instance that is constructed
        # with the provided arguments.
        #
        # Or we setup a new instance, and cache it

        if cache_id in _validator_cache:
            return _validator_cache[cache_id]

        # This means a cache miss

        # Make a sanitized version of the input arguments. These
        # can no longer have side effects that will change clone
        # behavior
        (args, kwargs) = tree_util.tree_unflatten(constructor_treedef, constructor_parameter_leaves)

        # Create the instance
        #
        # We specify the level in the inheritance
        # hierarchy directly as just above validator
        # to avoid issues

        instance = super(Validator, cls).__new__(cls)

        # Attach fields
        instance.next_validator = _next_validator
        instance.__args = args
        instance.__kwargs = kwargs
        instance.hash_value = cache_id

        # Cache it
        _validator_cache[cache_id] = instance

        # Return it. The patched __init__ (see __init_subclass__) will strip out the
        # feature _next_validator, then the user's init will take over.
        return instance

    #################
    # Define pytree logic so we can interface with broader jax
    #
    # This will include the flattening mechanism, and the unflattening mechanism
    ###############

    def make_node_spec(self) -> ValidatorNodeSpec:
        """
        Makes the node spec for the class.

        This represents the class in terms of
        constructor arguments and correct type, sans
        link information. It represents, in other words,
        the VALUE in a node.

        :return: A constructor spec
        """
        args, kwargs = self._get_constructor_parameters()
        return ValidatorNodeSpec(self.__class__,
                                 args,
                                 kwargs)

    def _get_nodespecs(self) -> List[ValidatorNodeSpec]:
        """
        Recursively executes a nodespec search
        and return a list in the order they
        were found

        NOTE: This will reverse the node order, but
        this is okay as we want that anyhow when
        we rebuild the nodes.
        """
        if self.next_validator is None:
            return [self.make_node_spec()]
        nodespec = self.next_validator._get_nodespecs()
        nodespec.append(self.make_node_spec())
        return nodespec

    def tree_flatten(self) -> Tuple[Any, Any]:
        """
        A validator list is defined in terms of the nodes that it has and
        the arguments that its constructors will receive

        :return: The flattened keys representing what was on
                 the node
        :return: The auxilary tree data, used to reconstruct
                 the node
        """
        return (), tuple(self._get_nodespecs())

    @classmethod
    def tree_unflatten(cls,
                       aux_data: Tuple[ValidatorNodeSpec, ...],
                       unused: Any
                       ) -> 'Validator':
        """
        Unflattens and reconstructs the original validator.

        Since the aux data will contain the nodes in reversed
        order, we just create and insert. This is efficient for
        a linked list.

        :param aux_data: The auxilary data
        :param unused: Completely unused here
        :return: The original validator
        """

        current_node = None
        for nodespec in aux_data:
            current_node = nodespec.type(
                *nodespec.args,
                _next_validator=current_node,
                **nodespec.kwargs
            )
        return current_node

    ###
    # Define linked list management mechanism. This includes magic
    # compositional shortcuts
    ###

    def fetch(self, item: int)->'Validator':
        """
        Fetch a particular validator node if it is
        available

        :param item:
        :return:
        """
    def append(self, validator: 'Validator') -> 'Validator':
        """
        Appends the validator provided onto the end of
        the linked list, and returns a totally independent
        list that does the same job.

        :param validator: The validator to append
        :return: A new validator node, with the new validator
                 appended to the end of the list.
        """
        args, kwargs = self._get_constructor_parameters()
        next_validator = self.next_validator
        if next_validator is not None:
            next_validator = next_validator.append(validator)
        else:
            next_validator = validator

        return self.__class__(*args,
                              _next_validator=next_validator,
                              **kwargs)

    def walk(self, f: Callable) -> Any:
        """
        Walks over the list, applying
        some function to the node
        and returning the result
        """
        yield f(self)
        if self.has_next:
            for item in self.next_validator.walk(f):
                yield item

    def __and__(self, other: 'Validator') -> 'Validator':
        """
        Supports chaining this validator with another validator using the `&` operator.
        This method adds the `other` validator to the end of the current chain.

        Example:
            validator = PositiveValidator() & EvenValidator()
            # Now `validator` will first check if the operand is positive, and then if it is even.

        :param other: Another instance of Validator to be chained.
        :return: The current instance with the `other` validator added to the end of the chain.
        """
        return self.append(other)

    ##########################
    # Define user overrides and methods access
    #
    # Methods going here are either things the user needs to impliment,
    # or validation mechanisms for if the user implemented their logic correctly
    #
    ########################



    def _execute_chain_predicate(self, **kwargs) -> bool:
        # A helper function that sanity checks
        # the users chain predicate
        try:
            outcome = self.chain_predicate(**kwargs)
        except Exception as err:
            raise create_subclass_code_did_not_run_exception(self, 'chain_predicate')
        if not isinstance(outcome, bool):
            raise create_subclass_code_returned_wrong_type_exception(self,
                                                                     code_feature='chain_predicate',
                                                                     required='bool',
                                                                     observed=outcome)
        return outcome

    @abstractmethod
    def predicate(self, operand: Any, **kwargs) -> bool:
        """
        Abstract method a subclass must implement. This
        should return true if validation is passed, and
        false otherwise

        :param operand: The operand that failed
        :param kwargs: The kwargs situation for the operand
        :return: A bool
        """

    def _execute_predicate(self, operand, **kwargs) -> bool:
        # A helper function that sanity checks
        # the users predicate
        try:
            outcome = self.predicate(operand, **kwargs)
        except Exception as err:
            raise create_subclass_code_did_not_run_exception(self, code_feature='predicate') from err
        if not isinstance(outcome, bool):
            raise create_subclass_code_returned_wrong_type_exception(self,
                                                                     code_feature='predicate',
                                                                     required='bool',
                                                                     observed=outcome)
        return outcome

    @abstractmethod
    def create_exception(self, operand: Any, **kwargs) -> Exception:
        """
        Abstract method that subclasses must implement to perform validation checks
        on the given operand.

        This method should return an exception to be raised and otherwise handled. It
        will only be triggered when the predicate fails.

        :param operand: The operand to be validated.
        :param kwargs: Additional keyword arguments that may be needed for validation.
                       These arguments are accessible to all validators in the chain.
        :return: A bool or bool array, and an exception to raise if failing
        """
        pass

    def _execute_create_exception(self, operand: Any, **kwargs) -> Exception:
        # A helper function that sanity checks and yells at
        # the user if they create an insane create exception
        try:
            exception = self.create_exception(operand, **kwargs)
        except Exception as err:
            raise create_subclass_code_did_not_run_exception(self, code_feature='create_exception') from err
        if not isinstance(exception, Exception):
            raise create_subclass_code_returned_wrong_type_exception(self,
                                                                     code_feature='create_exception',
                                                                     required='Exception',
                                                                     observed=exception)
        return exception

    def handle_exception(self, exception: Exception, **kwargs) -> Exception:
        """
        Handles an exception detected from further down in the
        validation chain. Does something with it. Must pass back
        an exception, but the exception need not be the same one.

        :param exception: The exception from downstream. You may log, throw,
                          or otherwise cause side effects. It is recommended
                          to log into kwargs if you do so
        :return: Either an error, in which case the error chain continues, or None,
                 which suppresses the error. You may even transform the original
                 error, and return the new error.
        """
        return exception

    def _execute_handle(self, exception: Exception, **kwargs):
        try:
            exception = self.handle_exception(exception, **kwargs)
        except Exception as err:
            raise create_subclass_code_did_not_run_exception(self, code_feature='handle_exception') from err
        if not isinstance(exception, Exception):
            raise create_subclass_code_returned_wrong_type_exception(self,
                                                                     code_feature='handle_exception',
                                                                     required='Exception',
                                                                     observed=exception)
        return exception

    ##################
    # Define logic used to actually perform valdiation
    #
    # The validation process is performed by passing
    # operand and kwargs from parent to children node,
    # and building an exception callback as we go. Shoudl
    # exception status be reached, the callback is called into
    # to handle the error.
    #
    # jax.lax.cond is usually used to handle branching to allow
    # for jit compatibility.
    #
    ###############

    ExceptionCallbackAlias = Callable

    ######################
    # Define validation pass and fail behavior
    #
    # Whether validation passes or fails, we will end up returning
    # the operand. This will hopefully ensure jit does not optimize it
    # away.
    #
    # However, on the failure branch, we execute a callback too
    # which is suppose to handle the error condition
    ##########

    @staticmethod
    def _base_case_passed(
            exception_callback: ExceptionCallbackAlias,
            success_callback: Callable[[Any, ...], None],
            operand: Any,
            **kwargs: Any
    ) -> Any:
        success_callback(operand, **kwargs)
        return operand

    def _execute_exception_callback(self,
                                    exception_callback: ExceptionCallbackAlias,
                                    operand: Any,
                                    **kwargs: Any
                                    ) -> Any:
        jax.debug.print("failed: {operand}", operand=operand)
        exception = self._execute_create_exception(operand, **kwargs)
        exception_callback(exception, **kwargs)

    def _base_case_failed(self,
                          exception_callback: ExceptionCallbackAlias,
                          success_callback: Callable[[Any, ...], None],
                          operand: Any,
                          **kwargs: Any
                          ) -> Any:

        jax.debug.callback(self._execute_exception_callback,
                           exception_callback,
                           operand,
                           **kwargs)
        return operand

    ########
    #
    # Several branch statements must be handled. This requires functional
    # usage of jax.lax.cond, and in places if statements. We need to handle
    # branches for:
    #
    # 1) Did validation pass? (cond statement)
    # 2) Did the chain predicate say continue? (cond statement)
    # 3) Is there a next_validator to check?
    #
    #####

    def _passed_branch(self,
                       exception_callback: ExceptionCallbackAlias,
                       success_callback: Callable[[Any, ...], None],
                       operand: Any,
                       **kwargs: Any
                       ) -> Any:
        """
        In the case where self validation passed, we need
        to decide whether to validate further entries in the
        chain, if they exist. We also need to eventually reach
        a base case

        :param callback: The growing callback
        :param operand: The operand to validate
        :param kwargs: The kwargs conditioning the validation
        :return:
        """
        # We have reached the passed base case if
        #
        # 1): There is no next entry.
        # 2): There is a next entry, but the chain predicate says not to check it
        #
        # Otherwise, we will call into the next validator in the chain,
        # which we now know exists

        if not self.has_next:
            return self._base_case_passed(exception_callback,
                                          success_callback,
                                          operand,
                                          **kwargs)
        chain_predicate = self._execute_chain_predicate(**kwargs)
        return jax.lax.cond(chain_predicate,
                            self.next_validator._validate,
                            self._base_case_passed,
                            exception_callback,
                            success_callback,
                            operand,
                            **kwargs
                            )

    def _validate(self,
                  exception_callback: ExceptionCallbackAlias,
                  success_callback: Callable,
                  operand: Any,
                  **kwargs) -> Any:
        """
        This class performs one portion of validation.

        This includes updating the callback stack if it exists,

        :param wrapped_exception_callback: The currently existing exception callback
        :param success_callback: The callback to call on success
        :param operand: The operand to check
        :param kwargs: The existing kwargs
        :return: The operand returned
        """

        @jax.tree_util.Partial
        def exception_callback_wrapper(exception: Exception, **kwargs: Any):
            # Ensures the node's handle function is called, then
            # pass the result further up the chain.

            exception = self._execute_handle(exception, **kwargs)
            exception_callback(exception, **kwargs)

        did_validation_pass = self._execute_predicate(operand, **kwargs)
        output = jax.lax.cond(did_validation_pass,
                              self._passed_branch,
                              self._base_case_failed,
                              exception_callback_wrapper,
                              success_callback,
                              operand,
                              **kwargs)
        return output

    def __call__(self, operand: Any, **kwargs) -> Any:
        """
        Executes the validation check on the given operand. If validation fails at any
        point in the chain, an exception is returned. Otherwise, None is returned,
        indicating successful validation.

        :param operand: The operand to be validated.
        :param kwargs: Additional keyword arguments for validation, passed to each validator
                       in the chain. When using a validator chain, one can pass in
                       kwargs which will be available whenever validate is called. This
                       is useful if, for instance, you need to specify batch size and will
                       not know it when creating a validator.
        :return: An Exception if validation fails at any point in the chain, None otherwise.
        """
        final_exception_callback = self.get_root_exception_callback()
        final_success_callback = self.get_success_callback()

        return self._validate(final_exception_callback,
                              final_success_callback,
                              operand,
                              **kwargs)
