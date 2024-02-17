from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Tuple
from jax import tree_util
from jax.tree_util import PyTreeDef

_validator_cache: Dict[int, 'Validator'] = {}

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

    Caching
    =======

    The validator class is very smart. It knows that
    """
    next_validator: 'Validator'
    hash_value: int
    __args: List[Any]
    __kwargs: Dict[str, Any]
    # Caching functions are used to store the parameters any subclasses
    # are called with such that we can make clones of them that are
    # independent of the original.

    def _store_constructor_parameters(self, args, kwargs):
        self.__args = args
        self.__kwargs = kwargs
    def _get_constructor_parameters(self)->Tuple[List[Any], Dict[str, Any]]:
        return self.__args, self.__kwargs

    def _store_hash(self, hash: int):
        self.hash_value = hash

    @classmethod
    def _get_class_identifier(cls)->str:
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
        unique_representation.append(cls._get_class_identifier()) # Account for class type
        if next_validator is not None:
            # Account for chain of nodes leading up to this point
            unique_representation.append(hash(next_validator))
        return hash(tuple(unique_representation))

    def __init_subclass__(cls, **kwargs):
        # A small, but limited to the class, monkey patch
        #
        # We patch init anytime we subclass to transparently accept and remove
        # "_next_validator" from the parameters that will be reaching the
        # user's __function__. The value '_next_validator' is sometimes passed
        # along when building a class by methods within this parent related to maintaining
        # the linked list. It is consumed in __new__. However, user __init__ functions will
        #  never need to know about that, and so we remove it before they see it.
        #
        # This prevents the user from having to pass along a node hook
        # every time they setup a new validator.

        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            if "_next_validator" in kwargs:
                kwargs.pop("_next_validator")
            original_init(self, *args, **kwargs)
        cls.__init__ = __init__

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

        # Cache it
        _validator_cache[cache_id] = instance

        # Return it. The patched __init__ (see __init_subclass_) will strip out the
        # feature _next_validator, then the user's init will take over.
        return instance

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

        kwargs = kwargs.copy()
        kwargs['_next_validator'] = next_validator

        return self.__class__(*args,
                              **kwargs)
    def chain_predicate(self, **kwargs):
        """
        A function that exists primarily to make suppressing
        errors easy, this function is called before checking the next
        node. By default it returns true - however, if it was to return
        false, the next node in the chain would not be checked

        :param kwargs: Any mutable context kwargs
        :return: A bool. True to continue the chain, false to break
        """
        return True
    @abstractmethod
    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        """
        Abstract method that subclasses must implement to perform validation checks
        on the given operand.

        This method should return None if the validation passes, or an Exception
        with an appropriate error message if the validation fails. All validators
        in the chain will have access to the `kwargs` argument, allowing for passing
        runtime dependencies or additional information required for validation.

        :param operand: The operand to be validated.
        :param kwargs: Additional keyword arguments that may be needed for validation.
                       These arguments are accessible to all validators in the chain.
        :return: None if the operand passes validation, or an Exception if it fails.
        """
        pass
    def handle(self, exception: Exception, **kwargs)->Optional[Exception]:
        """
        Handles an exception detected from further down in the
        validation chain. You may consider doing many different
        actions within this mechanism.

        :param exception: The exception from downstream. You may log, throw,
                          or otherwise cause side effects. It is recommended
                          to log into kwargs if you do so
        :return: Either an error, in which case the error chain continues, or None,
                 which suppresses the error. You may even transform the original
                 error, and return the new error.
        """
        return exception

    def __hash__(self)->int:
        return self.hash_value
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
        exception = self.validate(operand, **kwargs)
        if exception is not None:
            return exception

        if self.next_validator is not None:
            exception = self.next_validator(operand, **kwargs)
            if exception is not None:
                exception = self.handle(exception, **kwargs)
            return exception
        return None

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
