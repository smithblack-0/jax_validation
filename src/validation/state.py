"""

The states module is responsible for storing,
centrally and in one place, things related to the global
state of the project or that otherwise are nonlocal.

This may include things like context, or caches.
"""



from typing import Tuple, Type, Any, Optional, Callable, Dict
from .types import Operand

import jax.lax
import jax.tree_util
from jax._src.tree_util import PyTreeDef

###
# Define module fields all in one place
#
##
class StateData:
    def __init__(self):
        self.caches: Dict[str, 'Cache'] = {}
        self.final_callback: Optional[Callable[[Exception, ...], None]] = None
        self.success_callback: Optional[Callable[[Operand, ...], None]] = None
state = StateData()

###
# Define caching structure
# and manipulation methods
###

class Cache:
    """
    Caching can be used to speed up computational
    construction and usage processes in several locations

    This is a class to let that happen.
    """
    def __init__(self):
        self.cache = {}

    def setup_key_in_cache(self, key: Tuple[Type[Any], Tuple[...], PyTreeDef]):
        """
        Uses the provided key to setup the appropriate cache entry
        """
        cls, flat_params, treedef = key
        args, kwargs = jax.tree_util.tree_unflatten(treedef, flat_params)
        instance = cls(*args, **kwargs)
        self.cache[key] = instance

    def no_op(self, key: Tuple[Type[Any], Tuple[...], PyTreeDef]):
        """ Does nothing. Exists for cond call"""
        pass

    @staticmethod
    def create_unique_id(cls: Type[Any],
                         *args: Any,
                         **kwargs: Any):
        """

        :param cls: The class being identified
        :param args: The args that will be fed to any constructor
        :param kwargs: The kwargs that will be fed to any constructor
        :return: A hashable key representing the class
        """
        # Creating a unique key can be done by flattening the tree structure using
        # tree flatten to get a unique treedef, then
        constructor_parameters = (args, kwargs)
        flat_constructor_parameters, constructor_treedef = jax.tree_util.tree_flatten(constructor_parameters)
        flat_constructor_parameters = tuple(flat_constructor_parameters)
        hashable_representation = (cls, flat_constructor_parameters, constructor_treedef)
        return hashable_representation
    def fetch(self, constructor, *args, **kwargs)->Any:
        cache_id = self.create_unique_id(constructor, *args, **kwargs)
        jax.lax.cond(cache_id in self.cache,
                     self.no_op,
                     self.setup_key_in_cache,
                     cache_id,
                     )
        return self.cache[cache_id]


def get_cache(name: str):
    """
    Fetches a cache associated with something.

    So long as this function is provided with the
    same name, you get the same cache.

    :param name: The name of the cache
    :return: The cache entity.
    """
    if name not in state.caches:
        state.caches[name] = Cache()
    return state.caches[name]

###
# Define final exception callback methods and also create
# a context manager for exception callback status
###

def set_exception_callback(callback: Optional[Callable[[Exception, ...], None]]):
    """
    This modifies the context callback 'final_exception_callback'. This callback is
    run after all other handle features have been executed and may, for instance,
    be used to point your exception into your custom logger.

    :param callback: A callback that accepts an exception and a kwarg collection, or None
    """
    #TODO: wrap in validator here?
    state.final_callback = callback

def get_exception_callback()->Optional[Callable[[Exception, ...], None]]:
    """
    Gets the currently existing exception callback, this is user usable,
    but is usually used by internal logic.

    :return: A callback that accepts an exception and kwarg collection, or None
    """
    return state.final_callback

class ExceptionCallbackContextManager:
    def __init__(self, new_callback: Optional[Callable[[Exception, ...], None]]):
        """
        Initialize the context manager with the new exception callback.

        :param new_callback: The new exception callback to set when entering the context.
        """
        self.new_callback = new_callback
        self.old_callback = None

    def __enter__(self):
        """
        Set the new exception callback and save the old one.
        """
        self.old_callback = get_exception_callback()  # Save the old callback
        set_exception_callback(self.new_callback)  # Set the new callback
        return self  # You can return anything here, or nothing

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore the old exception callback when exiting the context.

        :param exc_type: The exception type, if an exception was raised in the context.
        :param exc_val: The exception value, if an exception was raised.
        :param exc_tb: The traceback, if an exception was raised.
        """
        set_exception_callback(self.old_callback)  # Restore the old callback
        # Return False to propagate any exceptions that occurred in the context
        return False


###
# Define the global methods, and context manager,
# for handling success callbacks
###

def set_success_callback(callback: Optional[Callable[[Operand, ...], None]]):
    """
    This modifies the context callback 'final_success_callback' which is expected to
    be run when validation passes, in case you want to do something on validation success

    It may be none, or a callback

    :param callback: A callback accepting the Operand and **kwargs. Or none, to have no callback
    """
    #TODO: wrap in validator here?
    state.success_callback = callback

def get_success_callback():
    """
    This gets the current success callback context, whatever it might be
    :return: A callback accepting the Operand and **kwargs. Or none.
    """
    return state.success_callback

class SuccessCallbackContextManager:
    def __init__(self, new_callback: Optional[Callable[[Operand, ...], None]]):
        """
        Initialize the context manager with the new success callback.

        :param new_callback: The new success callback to set when entering the context.
        """
        self.new_callback = new_callback
        self.old_callback = None

    def __enter__(self):
        """
        Set the new success callback and save the old one.
        """
        self.old_callback = get_success_callback()  # Save the old callback
        set_success_callback(self.new_callback)  # Set the new callback
        return self  # You can return anything here, or nothing

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore the old success callback when exiting the context.

        :param exc_type: The exception type, if an exception was raised in the context.
        :param exc_val: The exception value, if an exception was raised.
        :param exc_tb: The traceback, if an exception was raised.
        """
        set_success_callback(self.old_callback)  # Restore the old callback
        # Do not suppress exceptions, if any occurred within the context
        return False
