import unittest
import jax
from typing import Any, Optional, Tuple, Callable
from src.validation.core import ValidatorException, Validator
from jax import numpy as jnp
from src.validation import patching

jax.config.update("jax_traceback_filtering","off")

SHOW_ERROR_MESSAGES = True

class ValidatorInitializationTests(unittest.TestCase):
    """
    Test validator initialization mechanisms, including that
    initialization is possible and that linked list mechanism are
    executing correctly.

    The constructor is extremely complex for this class,
    relying on interactions between __new__, __init_subclass__,
    and even __init__ to work.

    As a result, we need to test it quite thoroughly
    """
    class MockValidator(Validator):
        def predicate(self, operand, **kwargs)->bool:
            return True
        def create_exception(self, operand: Any, **kwargs) -> Optional[Exception]:
            # Satisfies abstract
            return Exception()

    class MockValidatorWithInit(Validator):
        def __init__(self, do_raise: str):
            # Test
            self.do_raise = do_raise
        def predicate(self, operand, **kwargs)->bool:
            return self.do_raise == "raise"

        def create_exception(self, operand: Any, **kwargs) -> Tuple[bool, Exception]:
            return Exception("This is an exception")

        def handle_exception(self, exception: Exception, **kwargs) ->Exception:
            print(exception)
            return Exception()
    def test_constructor_basic_init(self):
        # Test we can initialize when just overwriting validate
        self.MockValidator()

    def test_constructor_basic_under_jit(self):
        def make():
            return self.MockValidator()
        make = jax.jit(make)
        make()

    def test_constructor_linked(self):
        # Test when we initialize a constructor with a link it is succesful.
        #
        # Required for linked list manipulation
        # This also verifies init is properly patched
        first_validator = self.MockValidator()
        second_validator = self.MockValidator(_next_validator=first_validator)
        self.assertIs(second_validator.next_validator, first_validator)


class TestHashFunction(unittest.TestCase):
    """
    Test cases for the initialization hash function are located here.

    The hash function's purpose is to detect when a new object being
    requested is equivalent to an existing one, and reuse the existing code
    when setting up.

    To do this, we need to understand when they will be equivalent. From the
    perspective of the hash function, it needs to accomodate:

    1) The class being instanced
    2) The arguments to the class constructor
    3) The chain of nodes that are being linked to the node being created

    In practice this is going to mean:

    1) The hash needs to depend on the qualified name of the class being instanced
    2) The hash needs to depend on the flattened arguments being fed to the constructor
    3) The hash needs to depend

    """
    def test_hash_nonrandom(self):
        """Test the same arguments gets the same results"""

        class subclass(Validator):
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True
            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception()

        leaves, treedef = jax.tree_util.tree_flatten(((),{}))

        # Test that hashes are different

        hash_one = subclass._create_hash(treedef, leaves, None)
        hash_two = subclass._create_hash(treedef, leaves, None)
        self.assertEqual(hash_one, hash_two)
    def test_hash_distinguishes_subclasses(self):
        """ Test that the hash function can produce a result that is dependent on the subclass"""

        # Create two different example subclasses to test with.
        #
        # They will be different classes, but otherwise have the same behavior
        class subclass_one(Validator):
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True
            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception()

        class subclass_two(Validator):
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True
            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception()

        # Create example constructor arguments

        leaves, treedef = jax.tree_util.tree_flatten(((),{}))

        # Test that hashes are different

        hash_one = subclass_one._create_hash(treedef, leaves, None)
        hash_two = subclass_two._create_hash(treedef, leaves, None)
        self.assertNotEqual(hash_one, hash_two)
    def test_hash_differentiates_arguments_differing(self):
        """
        Test that the hash function will properly detect and respond when the arguments for the
        constructor are different.

        In such a case, the hash should be different as well.
        """
        class subclass_with_init(Validator):
            """ A simple mock example that just stores item."""
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True
            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception()
            def __init__(self, item: int):
                self.item = item

        # Test that the hash differs when the hash of the leaves
        # differs

        constructor_arguments_one, constructor_treedef = jax.tree_util.tree_flatten(((3), {}))
        constructor_arguments_two, _ = jax.tree_util.tree_flatten(((4), {}))

        hash_one = subclass_with_init._create_hash(constructor_treedef, constructor_arguments_one, None)
        hash_two = subclass_with_init._create_hash(constructor_treedef, constructor_arguments_two, None)
        self.assertNotEqual(hash_one, hash_two, "Hashes were equal, cannot tell between differing argumnts")
    def test_hash_differentiates_linked_chain(self):
        """
        Test that the hash function can tell when different
        children are put into place
        """
        # Create a subclass
        class subclass(Validator):
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True
            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception()

        # Test with no child nodes, and with a singular one
        instance = subclass()

        leaves, treedef = jax.tree_util.tree_flatten(((),{}))
        hash_one = subclass._create_hash(treedef, leaves, None)
        hash_two = subclass._create_hash(treedef, leaves, instance)
        self.assertNotEqual(hash_one, hash_two)
    def test_hash_leaf_error(self):
        """
        Test that a sane and informative error is provided to the
        user in the event one of the leaves cannot be hashed
        """
        class subclass(Validator):
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True
            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception()

        # Manually select the unhashable type of list
        class unhashable:
           __hash__ = None

        unhashable_issue = {"test": unhashable()}
        leaves, tree_def = jax.tree_util.tree_flatten(unhashable_issue, lambda x : isinstance(x, list))

        # See what happens when we attempt to hash it
        with self.assertRaises(ValidatorException) as err:
            subclass._create_hash(tree_def, leaves, None)
        if SHOW_ERROR_MESSAGES:
            print("Test for if leaf is unhashable provides a sane error")
            print(err.exception)
            print(err.exception.__cause__)

class TestLinkedList(unittest.TestCase):
    """
    Test the mechanisms of the linked lists

    This includes automated merging, and the various helper functions.
    """
    def test_link_by_and(self):
        raise NotImplementedError()
    def test_append(self):
        raise NotImplementedError()
    def test_insert(self):
        raise NotImplementedError()
    def test_fetch(self):
        raise NotImplementedError()
    def test_slice(self):
        raise NotImplementedError()
    def test_walk(self):
        raise NotImplementedError()
    def test_str_representation(self):
        raise NotImplementedError()


class TestJitPytreeMechanisms(unittest.TestCase):
    class MockValidator(Validator):
        def predicate(self, operand, **kwargs)->bool:
            return True
        def create_exception(self, operand: Any, **kwargs) -> Optional[Exception]:
            # Satisfies abstract
            return Exception()

    class MockValidatorWithInit(Validator):
        def __init__(self,
                     do_raise: str):
            # Test
            self.do_raise = do_raise
        def predicate(self, operand, **kwargs)->bool:
            return self.do_raise != "raise"
        def create_exception(self, operand: Any, **kwargs) -> Tuple[bool, Exception]:
            return Exception("This test raised")

    class MockCallbackWithInit(Validator):
        def __init__(self, callback: Callable):
            self.callback = callback
        def predicate(self, operand: Any, **kwargs) ->bool:
            return True
        def create_exception(self, operand: Any, **kwargs) -> Exception:
            return Exception("This will never happen")
        def handle_exception(self, exception: Exception, **kwargs) ->Exception:
            self.callback(exception)
            return exception

    class MockValidatorWithKwargs(Validator):
        def predicate(self, operand, do_raise: bool, **kwargs)->bool:
            return jnp.logical_not(do_raise)
        def create_exception(self, operand: Any, do_raise: bool, **kwargs) -> Optional[Exception]:
            return Exception("Raise requested")

    def test_flatten_and_restore(self):
        validator = self.MockValidatorWithInit("raise")
        items, treedef = jax.tree_util.tree_flatten(validator)
        new_validator = jax.tree_util.tree_unflatten(treedef, items)
        self.assertIs(new_validator, validator)

    def test_chained_callback(self):
        class logging_observer:
            def __init__(self):
                self.errors = []
            def log(self, exception: Exception):
                self.errors.append(exception)
        class Logger(Validator):
            def __init__(self, observer: logging_observer):
                self.observer = observer
            def predicate(self, operand: Any, **kwargs) ->bool:
                return True

            def create_exception(self, operand: Any, **kwargs) -> Exception:
                return Exception("This should never happen")
            def handle_exception(self, exception: Exception, **kwargs) ->Exception:
                self.observer.log(exception)

        observer = logging_observer()
        validator = self.MockValidatorWithInit("raise")
        validator = Logger(observer, _next_validator = validator)
        try:
            validator(3)
        except Exception as err:
            print(err)
        print(observer.errors)


class ValidatorLinkedListTests(unittest.TestCase):
    class MockValidator(Validator):
        def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
            # Satisfies abstract
            return None
    class MockOtherValidator(Validator):
        def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
            # Satisfies abstract
            return None
    def test_append(self):

        to_append = self.MockOtherValidator()
        simple_list = self.MockValidator()
        longer_list = self.MockValidator() & self.MockValidator()

        new_simple_list = simple_list.append(to_append)
        self.assertTrue(new_simple_list is not simple_list)
        self.assertTrue(new_simple_list.next_validator is to_append)

        new_longer_list = longer_list.append(to_append)
        self.assertTrue(new_longer_list is not longer_list)
        self.assertIs(new_longer_list.next_validator.next_validator, to_append)
    def test_validator_chaining_magic(self):
        first_validator = self.MockValidator()
        second_validator = self.MockValidator()

        validator_chain = first_validator & second_validator
        self.assertIsInstance(validator_chain, self.MockValidator)
        self.assertIsInstance(validator_chain.next_validator, self.MockValidator)
class ValidateBehavior(unittest.TestCase):
    def test_kwargs_passed_through(self):
        """ Test that kwargs are cleanly passed, without change to every location"""
        kwarg = {"test" : 3}

        class KwargPassthrough(Validator):
            def validate(slf, operand: Any, **kwargs) -> Optional[Exception]:
                self.assertIn("test", kwargs)
                self.assertIs(kwargs["test"], 3)
            def handle(slf, exception: Exception, **kwargs) ->Optional[Exception]:
                self.assertIn("test", kwargs)
                self.assertIs(kwargs["test"], 3)
                return exception
            def chain_predicate(slf, **kwargs):
                self.assertIn("test", kwargs)
                self.assertIs(kwargs["test"], 3)
                return True

        class ProduceError(Validator):
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                return Exception()

        chain = KwargPassthrough() & KwargPassthrough() & ProduceError()
        chain(None, **kwarg)

    def test_validate_method(self):
        class MockValidator(Validator):
            def __init__(self, do_raise: bool):
                self.do_raise = do_raise
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                if self.do_raise:
                    return Exception()

        validator = MockValidator(True)
        self.assertIsInstance(validator(3), Exception)

        validator = MockValidator(False)
        self.assertIsNone(validator(3))
    def test_supression_predicate(self):

        # Define a chain of three things. The first should use the chain
        # predicate to shut down any further checking, and the second should throw
        # an error

        class ThrowError(Validator):
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                raise Exception()
        class SuppressError(Validator):
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                return None
            def chain_predicate(self, **kwargs):
                return False

        chain = SuppressError() & ThrowError()

        # Check that the error is in fact being prevented
        chain(3)

    def test_simple_chain_behavior(self):
        validator_chain = ThrowValidator() & PositiveValidator() & EvenValidator()
        with self.assertRaises(Exception) as context:
            validator_chain(-2)  # Should fail PositiveValidator
        self.assertIn("positive", str(context.exception))

        with self.assertRaises(Exception) as context:
            validator_chain(3)  # Should pass PositiveValidator but fail EvenValidator
        self.assertIn("even", str(context.exception))


class ValidatorCachingTests(unittest.TestCase):
    #TODO: Need more tests
    def test_validator_caching(self):
        first_instance = PositiveValidator()
        second_instance = PositiveValidator()
        self.assertIs(first_instance, second_instance)

    def test_validator_caching_with_different_params(self):
        first_instance = PositiveValidator()
        second_instance = EvenValidator()  # Different validator subclass
        self.assertIsNot(first_instance, second_instance)
    def test_caching_with_chaining(self):
        chain_one = PositiveValidator() & PositiveValidator() & PositiveValidator()
        chain_two = PositiveValidator() & PositiveValidator() & PositiveValidator()
        self.assertIs(chain_one, chain_two)
    def test_merge_chains(self):
        chain_one = PositiveValidator() & PositiveValidator()
        chain_two = chain_one & chain_one