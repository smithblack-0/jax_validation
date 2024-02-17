import unittest
from typing import Any, Optional

from src.validation.core import Validator

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

class ThrowValidator(Validator):
    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        return None
    def handle(self, exception: Exception, **kwargs) ->Optional[Exception]:
        raise exception

class WriteErrorToFile(Validator):

    def __init__(self, file_location: str, **kwargs):
        self.location = file_location

    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        return None

    def handle(self, exception: Exception, **kwargs) ->Optional[Exception]:
        # We write the exception to a file, alongside the
        # time.
        with open(self.location, mode="a") as f:
            f.write(str(exception))
        return exception

class SuppressErrors(Validator):
    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        return None
    def chain_predicate(self, suppress_errors: bool, *kwargs):
        return not suppress_errors

class ThrowIfNotSuppressed(Validator):
    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        raise RuntimeError()

class ValidatorInitializationTests(unittest.TestCase):
    def test_constructor_basic_init(self):
        positive_validator = PositiveValidator()
        self.assertIsInstance(positive_validator, PositiveValidator)
    def test_constructor_linked(self):
        even_validator = EvenValidator()
        positive_then_even = PositiveValidator(_next_validator=even_validator)
        self.assertIsInstance(positive_then_even.next_validator, EvenValidator)
    def test_constructor_overridden_init(self):
        write_to = WriteErrorToFile("errors.txt")
        self.assertTrue(write_to.location == "errors.txt")




class ValidatorChainingTests(unittest.TestCase):
    def test_validator_chaining(self):
        validator_chain = PositiveValidator() & EvenValidator()
        self.assertIsInstance(validator_chain, PositiveValidator)
        self.assertIsInstance(validator_chain.next_validator, EvenValidator)

    def test_chaining_two_lists(self):
        # Make sure we are merging the right way,
        # vs dropping or inserting.
        chain_one = PositiveValidator() & EvenValidator()
        chain_two = EvenValidator() & PositiveValidator()
        chain_composite = chain_one & chain_two

        current = chain_composite
        self.assertIsInstance(current, PositiveValidator)
        current = current.next_validator
        self.assertIsInstance(current, EvenValidator)
        current = current.next_validator
        self.assertIsInstance(current, EvenValidator)
        current = current.next_validator
        self.assertIsInstance(current, PositiveValidator)



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

    def test_supression_predicate(self):

        class ProduceError(Validator):
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                return Exception()

        class ThrowError(Validator):
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                return None
            def handle(self, exception: Exception, **kwargs) ->Optional[Exception]:
                raise exception

        class SuppressErrors(Validator):
            def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
                return None
            def chain_predicate(self, suppress_errors: bool, **kwargs):
                return not suppress_errors

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