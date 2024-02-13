from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict, Union


class TensorValidator(ABC):
    """
    Abstract base class for tensor validation. Validators can be chained together
    to perform a series of validation checks on a tensor.
    """

    def __init__(self, next_validator: Optional['TensorValidator'] = None):
        """
        Initializes the validator, optionally with a next validator to form a chain.

        :param next_validator: The next validator in the chain, if any.
        """
        self.next_validator = next_validator

    @abstractmethod
    def validate(self,
                 operand: Any,
                 **kwargs
                 ) -> Optional[Any]:
        """
        Validates the given operand according to specific criteria. You
        should return None if successful

        :param operand: The tensor to be validated.
        :param kwargs: Additional keyword arguments that may be needed for validation.
        :return: Nothing if successful. If unsuccessful, return arguments you can
                 use to build the message and the exception.
        """
        pass

    @abstractmethod
    def make_message(self,
                     operand: Any,
                     arguments: Any,
                     context_string: str,
                     **kwargs) -> str:
        """
        Generates an error message for a validation failure.

        :param operand: The tensor that failed validation.
        :param arguments: Any arguments stored during validate
        :param context_string: A context string providing additional information about the validation context.
        :param kwargs: Additional keyword arguments that may be relevant for generating the error message.
        :return: A string containing the error message.
        """
        pass

    @abstractmethod
    def make_exception(self,
                       message: str,
                       arguments: Any) -> Exception:
        """
        Creates an exception object from the given error message.

        :param message: The error message.
        :param arguments: Any arguments returned by validate.
        :return: An Exception object containing the error message.
        """
        pass

    def __call__(self, operand: Any, **kwargs) -> Optional[Exception]:
        """
        Executes the validation check on the given operand. If validation fails, an exception is returned.
        Otherwise, None is returned, indicating successful validation.

        :param operand: The tensor to be validated.
        :param kwargs: Additional keyword arguments for validation.
        :return: An Exception if validation fails, None otherwise.
        """

        if self.validate(operand, **kwargs) is not True:
            arguments = self.validate(operand, **kwargs)
            message = self.make_message(operand, arguments,  "Validation failed", **kwargs)
            exception = self.make_exception(message)
            return exception

        if self.next_validator is not None:
            return self.next_validator(operand, **kwargs)
        return None

    def __and__(self, other: 'TensorValidator') -> 'TensorValidator':
        """
        Chains this validator with another validator.

        :param other: Another instance of TensorValidator to be chained.
        :return: The current instance with the other validator added to the chain.
        """
        if self.next_validator is None:
            self.next_validator = other
        else:
            current = self.next_validator
            while current.next_validator is not None:
                current = current.next_validator
            current.next_validator = other
        return self

class PassthroughValidator(TensorValidator):
    """
    Performs no validation, just transparently passes the chain through
    """
    def validate(self,
                 operand: Any,
                 **kwargs
                 ) -> Union[bool, Any]:
        return True
    def make_message(self,
                     operand: Any,
                     arguments: Any,
                     context_string: str,
                     **kwargs) -> str:
        raise RuntimeError("This should never happen")
    def make_exception(self,
                       message: str,
                       arguments: Any) -> Exception:
        raise RuntimeError("This should never happen")
