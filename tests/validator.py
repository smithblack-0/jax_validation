import unittest
import jax
from typing import Any, Optional, Tuple, Callable
from src.validation.validator import Validator
from jax import numpy as jnp
from src.validation import patching

jax.config.update("jax_traceback_filtering","off")

class ValidatorInitializationTests(unittest.TestCase):
    """
    Test various validator mechanisms.

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

    class MockValidatorWithKwargs(Validator):
        def predicate(self, operand, do_raise: bool, **kwargs)->bool:
            return jnp.logical_not(do_raise)
        def create_exception(self, operand: Any, do_raise: bool, **kwargs) -> Optional[Exception]:
            return Exception("Raise requested")

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