import jax
import builtins
from typing import Tuple, Any, Type, List
from dataclasses import dataclass

@dataclass(frozen=True)
class ExceptionRepresentation:
    type: Type[Exception]
    args: List[Any]
    def __hash__(self):
        return hash(self.type)
    def build(self)->Exception:
        return self.type(*self.args)



already_registered = set()
def register_exception(exception: Exception):
    """
    Modifies jax's understanding of what an error is so
    that if jax goes to script an error, it returns a placeholder
    that can be passed around internally.

    :param exception: The exception to register and prepare
    """
    if exception in already_registered:
        return None
    def flatten_error(exception: Exception)->Tuple[Any, ExceptionRepresentation]:
        representation = ExceptionRepresentation(type(exception), exception.args)
        return (), representation

    def unflatten_error(auxilary: ExceptionRepresentation, flatten: Any)->Exception:
        return auxilary.build()

    jax.tree_util.register_pytree_node(exception, flatten_error, unflatten_error)
    already_registered.add(exception)

# Patch all default errors to be jittable
#for name in dir(builtins):
#    item = getattr(builtins, name)
#    if isinstance(item, type) and issubclass(item, BaseException):
#        register_exception(item)
