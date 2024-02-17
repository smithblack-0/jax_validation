
from jax import numpy as jnp
from src.validation.core import Validator
from typing import Optional, Any
from datetime import datetime

###
# Define operators to work on tensors. These will directly
# check the operands
##

class Probability(Validator):
    """
    Validates that the incoming operand is a probability
    """
    def __init__(self):
        pass
    def validate(self,
                 operand: jnp.ndarray,
                 **kwargs)->Optional[Exception]:
        if jnp.any(operand > 1.0):
            return ValueError("Probability in tensor exceed 1 on some elements")
        elif jnp.any(operand < 0.0):
            return ValueError("Probability in tensor is less than 0 on some elements")
        return None
class Floating(Validator):
    """
    Validates that the incoming operand is of
    """
    def validate(self, operand: jnp.ndarray, **kwargs) -> Optional[Exception]:
        if not jnp.issubdtype(operand.dtype, jnp.floating):
            return TypeError("Tensor was found to not be floating")

class ZeroOrGreater(Validator):
    def validate(self, operand: jnp.ndarray, **kwargs) -> Optional[Exception]:
        if jnp.any(operand < 0):
            return ValueError("Tensor was found to have values less than zero")
class Dtype(Validator):
    """
    Validates the incoming operand is a particular dtype
    """
    def __init__(self, dtype: jnp.dtype):
        self.dtype = dtype
    def validate(self, operand: jnp.ndarray, **kwargs) -> Optional[Exception]:
        if operand.dtype != self.dtype:
            return TypeError(f"Tensor wrong type: Expected type {self.dtype}, got {operand.dtype}")
class BatchShape(Validator):
    """
    Validates that the first dimension matches batch shape

    'batch_shape' = int must be provided as a kwarg
    """
    def validate(self, operand: jnp.ndarray, batch_shape: int, **kwargs) -> Optional[Exception]:
        if operand.shape[0] != batch_shape:
            return TypeError(f"Expected batch of shape '{batch_shape}', but got {operand.shape[0]}")

##
# Define meta operators.
#
# These will modify how the error is raise, or even
# if an error is raised at all
##

class PassthroughValidator(Validator):
    """
    Does nothing when validating, since
    other functions are going to be
    the relevant bits
    """
    def validate(self, operand: Any, **kwargs) -> Optional[Exception]:
        return None

class SuppressErrorsWhenFlagged(PassthroughValidator):
    """
    Allows for suppressing errors if the suppress_errors bool
    is true. This prevents any further validation from occurring
    """
    def __init__(self, suppress_errors: bool):
        self.suppress_errors = suppress_errors
    def chain_predicate(self, suppress_errors: bool, **kwargs):
        return not self.suppress_errors

class WriteErrorToFile(PassthroughValidator):
    """
    Writes the string representation of
    an error to the indicated file
    """
    def __init__(self, file_location: str, **kwargs):
        self.location = file_location

    def handle(self, exception: Exception, **kwargs) ->Optional[Exception]:
        # We write the exception to a file, alongside the
        # time.
        with open(self.location, mode="a") as f:
            time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            msg = time + "\n" + str(exception) + "\n"
            f.write(msg)

        # Notice how we return the exception. This continues the
        # process so earlier entries can also go off. We can also
        # not return anything, and completely stop error processing
        return exception


class ThrowOnError(PassthroughValidator):
    """
    Throws any encountered exception
    """
    def handle(self, exception: Exception, **kwargs) ->Optional[Exception]:
        raise exception

##
# Now lets see this all in action
#
# Start by mocking up bespoke integration. We would normally
# be loading the configuration options from a config somewhere,
# but we will let them be parameters for now

file_name = "errors.txt"
suppress_errors = False
def make_bespoke_terminal()->Validator:
    """
    Simulates loading bespoke dependencies.
    :return: A terminal chain.
    """
    bespoke_terminal = SuppressErrorsWhenFlagged(suppress_errors)
    bespoke_terminal = bespoke_terminal & ThrowOnError() & WriteErrorToFile(file_name)
    return bespoke_terminal
def cap_validator_chain(validator: Validator)->Validator:
    """
    Caps a validation chain with a common terminal node for
    prettyfing, logging, raising, and general bespoke integration

    :param validator: Validator to merge
    :param supress_errors: Normally, you would get this from a config, but we are
           making it a parameter for now
    :return: The completed validator
    """
    # We do not worry about rebuilding each time.
    # Caching will take care of that
    #
    # Also, caching means inserting and appending to the linked
    # list will behave about the same after the first run

    return make_bespoke_terminal() & validator
def make_probability_validator()->Validator:
    probability_validator = Floating() & BatchShape() & Probability()
    probability_validator = cap_validator_chain(probability_validator)
    return probability_validator


def make_counter_validator()->Validator:
    counter_validator = Dtype(jnp.int32) & ZeroOrGreater()
    counter_validator = cap_validator_chain(counter_validator)
    return counter_validator

# Lets do some testing!
batch_size = 2
valid_probability_array = jnp.array([[0.3, 0.6, 0.8],[0.2, 0.4, 1.0]])
valid_counter = jnp.array([[0, 2, 4]])

wrong_probability = jnp.array([[0.3, 0.6, 1.2],[0.2, 0.4, 1.0]])
bad_counter = jnp.array([[-2, 3.4]])

probability_validator = make_probability_validator()
counter_validator = make_counter_validator()

probability_validator(valid_probability_array, batch_shape=batch_size)
counter_validator(valid_counter)

# These should fail
try:
    probability_validator(wrong_probability, batch_shape =batch_size)
except Exception as err:
    print(err)

try:
    counter_validator(valid_counter)
except Exception as err:
    print(err)
