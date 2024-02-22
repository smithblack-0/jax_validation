import textwrap

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Tuple, Callable, Type

import jax.tree_util
from jax import tree_util
from jax.tree_util import PyTreeDef
from jax import numpy as jnp

_validator_cache: Dict[int, 'Validator'] = {}

@dataclass(frozen=True)
class ValidatorNodeSpec:
    type: Type['Validator']
    args: Any
    kwargs: Any

class ValidatorException(Exception):
    """
    Raised when something goes wrong during execution
    of a validator that is not connected to the origina
    validation target.

    An example would be a user misidentifying an operand
    """
    def __init__(self,
                 subclass: Type,
                 validation_feature: str,
                 details: str
                 ):

        header = f"An issue occurred while trying to execute subclass-provided method '{validation_feature}\n'"
        body = textwrap.dedent(details)
        tail = f"This occurred on validator of type '{type(subclass)}'"
        message = header + "\n" + body + "\n" + tail

        message = textwrap.indent(message, "    ")
        message = "A ValidatorException occurred while trying to validate input: \n" + message

        super().__init__(message)
        self.message = message

class SubclassDidNotRunException(ValidatorException):
    def __init__(self,
                 subclass: Type,
                 validation_feature: str,
                 ):
        details = f"""\
        The provided code did not run successfully. It is possible that
        your code was malformed, you did not provide needed kwargs, or even
        that jit is acting up
        """
        super().__init__(subclass, validation_feature, details)

class SubclassReturnedWrongTypeException(ValidatorException):
    def __init__(self,
                 subclass: Type,
                 validation_feature: str,
                 required: str,
                 observed: Any
                 ):
        details = f"""\
        The provided code ran just fine, however it did not return the 
        correct type. The requred type was '{required}', but what was
        observed was '{type(observed)}
        """
        super().__init__(subclass, validation_feature, details)

class NullException(Exception):
    pass


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
    by jax.tree_util.flatten, and that has hashable leaves. Also, you cannot
    expect to rely on side effects from features passed into __init__.

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


    next_validator: 'Validator'
    hash_value: int
    list_length: int
    __args: List[Any]
    __kwargs: Dict[str, Any]

    @property
    def has_next(self)->bool:
        return self.next_validator is not None

    @property
    def num_validators(self)->int:
        if not self.has_next:
            return 1
        return self.next_validator.num_validators + 1

    #################
    # Define initialization routines.
    #
    # The majority of the complexity of the class is the initialization routines.
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
    def _store_constructor_parameters(self, args, kwargs):
        self.__args = args
        self.__kwargs = kwargs
    def _get_constructor_parameters(self)->Tuple[List[Any], Dict[str, Any]]:
        return self.__args, self.__kwargs
    @classmethod
    def _get_unique_class_identifier(cls)->str:
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
        unique_representation = [constructor_treedef, tuple(constructor_leaves)] # Account for constructor arguments
        unique_representation.append(cls._get_unique_class_identifier()) # Account for class type
        if next_validator is not None:
            # Account for chain of nodes leading up to this point
            unique_representation.append(hash(next_validator))
        return hash(tuple(unique_representation))
    def _store_hash(self, hash: int):
        self.hash_value = hash

    def __hash__(self)->int:
        return self.hash_value

    def _check_list_length(self)->int:
        if self.next_validator is None:
            return 1 # base case
        return self.next_validator._check_list_length() + 1
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
        # We also register the subclass with jit.

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
        instance._store_constructor_parameters(args, kwargs)
        instance._store_hash(cache_id)
        instance.list_length = instance._check_list_length()

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


    def make_node_spec(self)->ValidatorNodeSpec:
        """
        Makes the node spec for the class.

        This repreesents the class in terms of
        constructor arguments and correct type, sans
        nodes.
        :return: A constructor spec
        """
        args, kwargs = self._get_constructor_parameters()
        return ValidatorNodeSpec(self.__class__,
                                 args,
                                 kwargs)
    def _get_nodespecs(self)->List[ValidatorNodeSpec]:
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

    def tree_flatten(self)->Tuple[Any, Any]:
        """
        A validator list is defined in terms of the nodes that it has and
        the arguments that its constructors will receive

        :return: The flattened keys representing what was on
                 the node
        :return: The auxilary tree data, used to reconstruct
                 the node
        """
        return () , tuple(self._get_nodespecs())

    @classmethod
    def tree_unflatten(cls,
                       aux_data: Tuple[ValidatorNodeSpec,...],
                       unused: Any
                       )->'Validator':
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
    def append(self, validator: 'Validator')->'Validator':
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
                              _next_validator = next_validator,
                              **kwargs)
    def walk(self, f: Callable)->Any:
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

    def chain_predicate(self, **kwargs)->bool:
        """
        A function that exists primarily to make suppressing
        errors easy and efficient. This function is called before checking the next
        node. By default it returns true - however, if it was to return
        false, the next node in the chain would not be checked

        :param kwargs: Any mutable context kwargs
        :return: A bool. True to continue the chain, false to break
        """
        return True

    def _execute_chain_predicate(self, **kwargs)->bool:
        # A helper function that sanity checks
        # the users chain predicate
        try:
            outcome = self.chain_predicate(**kwargs)
        except Exception as err:
            raise SubclassDidNotRunException(self, validation_feature='chain_predicate') from err
        if not isinstance(outcome, bool):
            raise SubclassReturnedWrongTypeException(self,
                                                     validation_feature='chain_predicate',
                                                     required='bool',
                                                     observed=outcome)
        return outcome

    @abstractmethod
    def predicate(self, operand: Any, **kwargs)->bool:
        """
        Abstract method a subclass must implement. This
        should return true if validation is passed, and
        false otherwise

        :param operand: The operand that failed
        :param kwargs: The kwargs situation for the operand
        :return: A bool
        """

    def _execute_predicate(self, **kwargs) -> bool:
        # A helper function that sanity checks
        # the users predicate
        try:
            outcome = self.predicate(**kwargs)
        except Exception as err:
            raise SubclassDidNotRunException(self, validation_feature='predicate') from err
        if not isinstance(outcome, bool):
            raise SubclassReturnedWrongTypeException(self,
                                                     validation_feature='predicate',
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

    def _execute_create_exception(self, operand: Any, **kwargs)->Exception:
        # A helper function that sanity checks and yells at
        # the user if they create an insane create exception
        try:
            exception = self.create_exception(operand, **kwargs)
        except Exception as err:
            raise SubclassDidNotRunException(self, validation_feature='create_exception') from err
        if not isinstance(exception, Exception):
            raise SubclassReturnedWrongTypeException(self,
                                                     validation_feature='create_exception',
                                                     required='Exception',
                                                     observed=exception)
        return exception

    def handle_exception(self, exception: Exception, **kwargs)->Exception:
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
            raise SubclassDidNotRunException(self, validation_feature='handle_exception') from err
        if not isinstance(exception, Exception):
            raise SubclassReturnedWrongTypeException(self,
                                                     validation_feature='handle_exception',
                                                     required='Exception',
                                                     observed=exception)
        return exception

    ##################
    # Define logic used to operate the class
    ###############

    def _wrap_exception_callback(self,
                                 exception_callback: Callable,
                                 **kwargs
                                 ):
        """
        Wraps the exception callback passed in a closure that
        will first execute my handling function, then pass the
        result back upstream

        :param exception_callback: The current execution callback
        :param kwargs: Any kwargs of note
        :return: The new execution callback, with my callback inserted
        """
        my_callback = jax.tree_util.Partial(self._execute_handle, self, **kwargs)
        def closure(effects_token, exception: Exception):
            jax.debug.callback(my_callback, exception)
            jax.debug.print(str(exception) + str(effects_token))
            return exception_callback(effects_token, exception)
        return closure


    def _run_validation(self,
                        operand: Any,
                        exception_callback: Any,
                        **kwargs
                        ):
        def continue_met_branch():
            return self.next_validator._run_validation(effects_token, operand, exception_callback, **kwargs)

        def continue_branch():
            # Continue validating if there are further elements in the chain
            if self.has_next:
                continue_predicate = self._execute_chain_predicate(**kwargs)
                jax.lax.cond(continue_predicate,
                             continue_met_branch,
                             lambda : None
                             )

        def exception_branch():
            # Call into the exception callback if an exception state has been reached
            exception = self._execute_create_exception(operand, **kwargs)
            return exception_callback(effects_token, exception)

        exception_callback = self._wrap_exception_callback(exception_callback)
        predicate = self.predicate(operand, **kwargs)
        return jax.lax.cond(predicate,
                     continue_branch,
                     exception_branch
                     )

    def _check_children(self, operand, **kwargs):
        """
        Checks child nodes if possible, and if not
        returns the base case of zero
        """
        if self.has_next:
            return self.next_validator._check_error_code(operand, **kwargs)
        return 0

    def _handle_chain_predicate(self, operand, **kwargs):
        """
        If here, the validation should have passed. Either
        return 0 if chain predicate is false, indicating
        base case, or go into check_childrein if true
        """
        return jax.lax.cond(self.chain_predicate(**kwargs),
                            self._check_children,
                            lambda operand, **kwargs : 0,
                            operand,
                            **kwargs
                            )

    def _check_error_code(self, operand, **kwargs)->int:

        validation_passed = self.predicate(operand, **kwargs)
        error_code = jax.lax.cond(~validation_passed,
                            lambda operand, **kwargs: 1, #Base case: Failed
                            self._handle_chain_predicate, # Continue: Succeess
                            operand,
                            **kwargs
                            )

        # Increment by one each time you go up a node. If the validation failed, it means
        # we are just passing back the error so we do not increment.
        error_code = jax.lax.select(validation_passed & (error_code > 0), error_code + 1, error_code)
        return error_code


        def continue_cond(state: Tuple[int, int, bool]):
            _, _, should_we_continue = state
            return should_we_continue

        def update_function(state: Tuple[int, int, bool]):
            node_index, error_code, _ = state

            predicate = self.fetch(node_index)(operand, **kwargs)
            chain_predicate = self.fetch(node_index)(operand, **kwargs)
            has_next = self.fetch(node_index).has_next

            error_code = jax.lax.select(predicate, error_code, node_index)
            should_continue = predicate & chain_predicate & has_next

            return node_index + 1, error_code, should_continue

        _, error_code,  = jax.lax.while_loop(continue_cond,
                                              update_function,
                                              (0, -1, True)
                                              )
        return error_code + 1



    def __call__(self, operand: Any, **kwargs) -> Optional[Exception]:
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
        error_code = self._check_error_code(operand, **kwargs)
        print(error_code)
        self._run_validation(operand, error_stack, lambda exception: None, **kwargs)
        print(error_stack)
        return error_stack