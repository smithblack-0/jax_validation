from patching import Validator
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Dict, Union

class TensorValidator(Validator, ABC):
    """
    An abstract base class dedicated to directly
    validating tensors. It is what most users
    will be implementing.
    """

    @abstractmethod
    def predicate(self, operand: Any, **kwargs) -> bool:
        """
        A predicate for validation. When it fails, the
        validation is considered to have failed.

        :param operand: The operand to validate
        :param kwargs: Any kwargs that are needed for validation, like shape or batch size
        :return: A bool. True means pass, false means failed
        """

    @abstractmethod
    def exception_factory(self,
                          operand: Any,
                          **kwargs) -> Exception:
        """
        A method that is called to manufature an exception
        once the predicate has failed. It should provide an informative
        error of some sort.

        :param operand: The operand which failed
        :param kwargs: The kwargs for the call
        :return: An exception type
        """

    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        if not self.predicate(operand, **kwargs):
            return self.exception_factory(operand, **kwargs)
        return None