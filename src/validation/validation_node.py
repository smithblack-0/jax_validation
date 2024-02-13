from typing import Any, Optional
from abc import ABC, abstractmethod

class Validator(ABC):
    """
    Base interface for a validation operator
    """


    def __init__(self, next_validator: Optional['ValidationNode'] = None):
        self.next = next_validator

    @abstractmethod
    def __call__(self, operand: Any, **kwargs)->Optional[Exception]:
        pass

