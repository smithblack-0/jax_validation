## Introduction

Hello All.

I have been developing a verification architecture and corresponding library that 

I got started on this whole journey partly because I was frustrated with how primitive the jax validation mechanisms are for value-based cases (jax.jit, WHY do you not automatically stage checkify!!!?) and partly as an exercise in how to build a really solid validation system built specifically for tensors that is production quality.

While I admit at first I thought I would likely be keeping it to myself, the end
product is proving to be sophisticated enough I would like to see
if the broader community might be interested in incorporating it
in some manner into jax itself, or into some of the 
lists of resources, when it is done.

## What is the architecture

Currently, I have an elegant way to reuse 
validation operands and support for most professional
error situations that I can think of.

Basically, you define a validation operator using the 
chain-of-responsibility pattern, so that each node in the 
chain can be responsible for validating one thing, or maybe
handling an exception from further down the chain. Then,
you link them together. This is done using magic methods
and a straightforward architecture like below:

```python

# Presuming we have defined operators ValidateIsFloating
# and ValidateIsProbability elsewhere
#
# This will define a chain of responsibility that calls
# into first ValidateIsFloating, then ValidateProbability

validate_probability = ValidateIsFloating() & ValidateIsProbability()

# Note, however, this will not raise on error. Instead, it
# returns an error

error = validate_probability(3.2)
if error:
    raise error
```

Once invoked, errors are not actually raised, but 
passed back up the chain of responsibility. This means you
can do some clever things, like changing the behavior to log
or throw, just by swapping out the head of the chain. Or
even just define a custom terminal node to integrate into your
existing framework.

```python

# Lets suppose we have LogToFile and ThrowIfError
# defined somewhere upstream. We also have validate
# probability from before. 

# We can create a logging version of it like so:

log_probability_errors = LogToFile("errors.txt") & validate_probability

# Or a version that will throw

throw_probability_errors = ThrowIfError() & validate_probability

# To use it we can just call it

log_probability_errors(3)

# We could even log then throw

log_then_throw = ThrowIfError() & LogToFile("errors.txt") & validate_probability
```

Under the hood, the library sets up a chain of responsibility - basically
a linked list - with each node in the list responsible for
its own task. Nodes can be responsible for detecting and 
returning an exception, handling an exception from further down
the stack. Everything is built to operate in a functional
manner, meaning you will not break your linked lists using 
side effects. Additionally, to handle the performance hit
this can cause, everything is cached.  

Finally, if I am not misreading the situation, it can also assist
significantly with jit compatibility. The traditional issue is that 
you cannot raise errors on the values of tensors, as it creates divergent
jit paths. We can fix that. If you mark your error messages as, say
error._jit_incompatible, you can catch the error as it is going up and
choose to, say, print it to the console using jax.debug.print. That would
safely discharge the side effects and allow jax to continue, without needing
to explictly stage checkify.

## What is working, and what is planned?

### Working
Currently, the core validator is working and mostly
tested, although I have not done caching much before and 
need to figure out what tests to write for the hash function.


### Planned: Operator library

This means simple behavior, such as validating a singular
operand against many different cases, is supported. Also supported
is reusing previously defined Validator chains in any different context.
Everything is functional, so do not worry about side effects.

Not implemented yet is the basic operators library. I could use some feedback
here

* What are your common validation cases? 
* What bespoke integration do you commonly need to perform?
* Am I missing anything?

### Pytrees

Pytree support is a planned core part of the libraries

The whole reason I am working in jax rather than torch or tensorflow, in fact, is pytrees. 
So I have been somewhat frustrated that pytrees are so tricky to validate. You
basically have to hunt down and understand jax.tree_util.tree_map, and if you
want to validate that two trees are the SAME it gets a whole lot harder.

I do not like that. Instead, I plan to create a class called Schema
which can be made per batch or statically, and which itself contains
a pytree. You will be able to specify validator chains to run
for each leaf of the pytree. You will also be able to use it to,
for example, apply one validator chain across an entire 
pytree in "prefix tree broadcasting." 

Thankfully, I recently discovered that _src.tree_util under the 
source code has utilities for dong this already, and even for raising good error messages, 
so I will be importing a lot of the work from there. Since they are unit tested,
I think it should be fine. It also should be fairly fast, hopefully, as I
will be leaning on the optimized jax code.

## Can I have?

At the moment, the library is not shared on mypy, but the repository is located at
the following link if you want to poke around. Please note it is fairly unpolished 
at the moment:





## An example?

Lets consider the following spec

```markdown

Bespoke integration:
- You shall log any error you encounter to 'error.txt'
- You shall throw any error you encounter after logging it
- You shall allow for the suppression of validation. When suppressed,
  validation shall incur a minimal performance penalty.
  
Probabilities Tensor Validation
- You shall check that the batch shape is sane
- You shall check that the type is floating
- You shall check that the values are between 0 and 1
- You shall provide me with a function that can test this

Counter Tensor Validation:
- You shall check that the dtype is int32
- You shall check that the values are greater than or equal to zero
- You shall provide me with a function to test this

... etc, but that is enough for now.
```

 Lets see how the current library would implement that. This code is entirely
functional at the moment, and since the operators library is not in place
yet we have to define each validation operation ourself. That is not a bad thing,
though, as it shows the three main methods that can be overridden to influence
behavior

```python

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

## YOUR CODE STARTS HERE
#
# In practice, the above will likely be hidden away in a library
# somewhere, and you will be dealing with logic more like the below
#
# Though, if you ever need to extend the library, you certainly can

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
