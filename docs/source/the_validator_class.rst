Validators are immutable linked list nodes. Each node is designed to perform one validation
task, then ask the next node for it's task as well. The immutability is a somewhat
difficult property to maintain, but worth it as it significantly simplifies downstream
logic. It also means that if you build up a validation chain, you can safely apply it in
multiple places.

Lifecycle of a validator
========================

Conceptually, there are several phases to setting up and using
a validator. They might be categorized into:

    - definition: Subclass the validator, and define it in the first place, creating reusable
                  templates for a particular task
    - setup: Create particular instances of the validator, and fill in any init details
    - linkage: Link validators together into a chain using composition to achieve a particular task
    - usage: Call the validator


Defining and using a validator
==============================

Conceptually, there are three phases to a validators lifecycle. These are the definition
phase, the setup phase, and the usage phase.

The definition phase involves defining what we actually want the
validator to do. To do this, we must define abstract method
'validate', and return either none, when validation passes,
or an exception, if it fails.

Note that regardless on whether you plan to use them or not,
you still need to specify kwargs.

```
class PositiveValidator(Validator):
    # This simple class works without an init function
    def validate(self, operand, **kwargs):
        if operand > 0:
            return None
        return Exception("Operand must be positive")
```

Defining a validator using init
---------------------------------

Setting up an __init__ call is optional, and may not be needed
for simple validators. However, some validators, that should be setup once
and used many times, may benefit from using an init. Lets see an example

```
class IsDtype(Validator):
    #This sets up a threshold on the fly
    def __init__(self, dtype: jnp.dtype):
        self.dtype = dtype
    def validate(self, operand, **kwargs):
        if operand.dtype != self.dtype
            raise TypeError("Wrong Dtype")
```

Defining a validator using runtime mutables
-------------------------------------------

Sometimes, you may have features for which you do
not details of until you actually run the program.
Batch shape might be a good example.

Such cases are also covered. You can configure your validate
function to retrieve something out of kwargs, or accept a kwarg from
upstream. Note that *args are not allowed.

For example:

```
class EnsureBatchShape(Validator):
    def validate(self,
                operand: jnp.ndarray,
                batch_shape: int,
                **kwargs)->Optional[Exception]:
        if operand.shape[0] != batch_shape

setup and linkage
-----------------

Lets consider the three validators defined above, and one more validator
called throw. We will now examine how to use setup and linkage to construct
a validation chain.

First, for convenience, define throw:


- Statically define validator: Pass in the init arguments for this validator. These should never
  be things you will rely on side effects for
- Apply validation with runtime kwargs:
