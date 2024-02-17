Development notes
#################

Round 0: Validation considerations
===========================

What does it mean to validate something? What kinds of actions can occur? What
kinds of dependencies might exist? Lets consider this.

Validation occurs when you take some group of inputs, check some condition, and
output an issue if the condition is not met.

Types of validation
-------------------

Validation comes in several flavors.

- Operand validation: You check if values match probabilities, or maybe batch shape.
- Relationship validation: You check if tensors or PyTrees behave in some sort of manner
- Schema validation: You check if tensors or PyTrees match some sort of schema

What does validation operate on?
--------------------------------

Lets limit ourselves to PyTrees and jax tensors.

Actions during validation
-------------------------

- On success: Do nothing
- On failure: Put together insightful error message, raise error action.

Dependencies of validation
--------------------------

Validation may have various level of dependencies depending on what exactly
is being validated.

- Some features, such as what is a probability, could be defined once and used many times
- Some features, such as what is the correct batch shape, will either need to be defined
  on the fly or have information passed as arguments to work correctly
- Making good messages will almost certainly require string formatting or something similar.

Round 1: Example Use Cases
=================================

Ensure Probability
------------------

Case: Ensure incoming tensor is a probability
Actions: Ensure elements are between 0 and 1
Messaging: If not, emit message saying so, and error type. Might want to know what
           tensor's values were
Restrictions: Not strictly jit compatible when raised
Conclusions:

-   Need a boolean predicate, and a message system.
-   Mostly independent of batch-specific information

Ensure correct dtype:
---------------------

Case: Ensure incoming tensor is of a certain dtype
Actions: Ensure dtype
Messaging: Emit a message saying expected, and actual, dtypes
Restrictions: None
Conclusions:
-   Need to know information from the failed tensor when raising.
-   Mostly independent of batch specific information

Ensure matching batch shape
---------------------------

Case: Ensure tensor matches a batch shape
Actions: Have batch shape, compare tensor to it, ensure it matches
Messaging: Emit a message saying what went wrong and where the mismatch
           is if needed. Requires batch shape, tensor
Restrictions: None
Conclusions
-   Would have to know batch shape when validing. This means information has to be updated
    or managed between batches
-   Batch shape can change. This might mean it is a good idea to allow information to
    be moved around as arguments
-   It also might be possible to just rebuild the validation mechanism, somehow,
    each time you make a new batch.

Ensure matching static PyTree Schema
-----------------------------

Case: Ensure a pytree shape matches a static schema. Do not care about the content. C
Actions: Have some sort of static definition, see if right pieces are in right place in incoming
         pytree.
Messaging: Should raise message when schemas do not match, indicating what the problem is
           and where.
Restrictions: None
Conclusions:
-   The static case could be easily dealt with by making a validator to specifically
    handle the problem.
-   The dynamic case is interesting
-   The schema could be defined by making a pytree and handing it to a class or function

Ensure matching static PyTree Schema and dynamic details
--------------------------------------------------------

Case: Ensure a PyTree shape matches a static schema. Ensure the tensors contained
      match dynamic quantities such as correct batch shape or dtype.
Actions: Verify static schema match. Meanwhile, somehow apply case-specific
         actions to pytree leaves
Conclusions:
-   The static case could still be defined similar to above, however...
-   It might be useful to allow placing validators as leaves in the schema or something
    to associate, say, different dtypes with different leaves.
-   Highly dynamic features, such as validation for batch shape, may require arguments or
    redefining for every batch.
-   This can still be executed with a single targetted operand

Ensure two matching pytree are compatible
------------------------------------------------

Case: Ensure two different pytrees have the same tree structure.
Actions: Compare the two pytrees somehow, point out what is wrong
Conclusions:
    - Given the behavior outlined above, it might be best to
      dynamically create some sort of "schema" then allow it to be fed as a parameter
      into a validation mechanism.
    - There is some level of control over leaf schema behavior that will likely be needed.

Overall conclusions
-------------------

I conclude

- Architecture should support statically defined leaf definitions such as "probabilities"
- Architecture should support dynamically defined schemas and leaf restrictions
- Architecture should work around jit incompatibility
- Architecture should allow passthrough of extra information

Round 2: Core feature: Validator
======================

Validation Class: Validator

We define there to exist the following main
validation mechanism. It has specification as follows:

**General concept**

- Abstract
- Can accept and utilize kwargs and a target operand to target for validation. Will then
  perform validation
- Is a "Chain of responsibility" mechanism, meaning a one way
  linked list is being built internally.
- Validation override is an abstract method

**Specification**
- Contains constructor __new__:
    - Sets up method 'clone' that can be provided with 'next_validator' and
      will setup a node with the exact same parameters, but pointing to that
      next validator instead
    -
- Contains constructor __init__:
    - Constructor can be provided with another validation class
    - This will be stored to see to the chain of responsibility.
    - This is optional.
- Contain function called *validate*:
    - It indicates whether validation succeeds or fails
    - It is abstract - subclasses need to implement
    - Accepts operand, kwargs
    - Returns either None, if validation passes, or an Exception, if fialed.
- Contains function for __call__
    - Performs validation.
        - Validates self condition.
        - Calls into any children with operand, passing on kwargs.
    - Accepts operand, kwargs
    - Returns either an exception, or None if nothing goes wrong

**Manipulation pattern: Defining**

It is intended you define your validation mechanisms all at once
statically somewhere in your program, and pass arguments as needed.

For instance

```
class Probability(TensorValidator):
    .... your details here

class MatchBatchShape(TensorValidator):
    ... your details here
```

Then, later on when you need one, you go:

```
validator = Probability()
```

**Manipulation pattern: Chaining**

Multiple of these classes can also be linked together by means of basic
boolean operations. For example, lets suppose you have validators

-   Probability
-   MatchBatchShape

You want to make a new validation mechanism that checks both. You can go ahead
and use python magic methods like so:

```
validator = Probability()
validator = validator & MatchBatchShape()
```

The classes will then rebuild themselves to create a linked mechanism that first
executes probability, then MatchBatchShape

Round 3: Core Features: PyTreeValidator
=======================================

my tools from for validation of pytree are puny. In particular
    - I have the ability to flatten a tree and get a list of path, leafs. This is implemented in c++, fast
    - I have the ability to build my own walker that calls into jax's treedef. Slow
    - I do NOT have the ability to walk trees in parallel using my own predicates or logic
      under their framework.

I would like to be able to specify the pytree shape fairly naturally
    - Provide a pytree that the shape needs to match. Optionally, provide conditions
    - Statically build
    - Reveal shape at runtime.
    - Broadcasting - one node on a schema's Validator applies to many children


Two main types of issues can occur here. They are
    - The Pytree's shape is malformed.
    - The check for an individual leaf failed.

I would like to have detailed error messages
    - This includes information on where the nodes diverge.

So:
    - flatten both schema and operand trees
    - walk through children, and compare
    - Pass through validator checks, and provide useful error info on WHERE a pytree is malformed

schema
------

A schema is an object that contains a pytree with children of None or Validator. Downstream objects
will broadcast a schema if needed - this means that if the schema ends on a validator leaf, while the operand
pytree ends on a branch, all decendent children of that branch will see the same validator leaf applied to them.

- The pytree may be filled with any jax nodes, so long as they terminate with None or Validator
- The schema has a method .attach_head, that takes every child node and inserts the head at the front, like:
    new_node = head & node
- The schema has a method .attach_tail. It takes every node and inserts tail near the end. Like:
    new_node = node & tail

error messages
--------------

Structural issue messages:

- point out something is missing or different, and where it is
for example:

"Along branches: x/y/z

The following difference occurred between original and novel branch

                original,                  novel
                KeyNode("hi"),           Keynode("hi")
difference ->   KeyNode("potato")        Keynode("tomato")
difference ->   KeyNode("tomato")        Keynode("potato")

Getting this done means:

- Notice when the keypaths become incompatible
- Track down where the divergence happens
- Mockup children at that level, extract info
- Reveal this info in console.
condition-by-condition
----------------------

No issue:
    - The paths match. We apply the validator. It matches
Broadcasting:
    - The paths do not match. check if
Lets say we are walking the above. What can happen?
    - We compare the paths. They do not match.
    - We check for

Round 2: Core Features: Schema Class
====================================

The Schema class is responsible for presenting
and otherwise knowing how to handle and look at pytrees
in a jax environment. PyTreeValidators, meanwhile, can anchor and consume
schema specifications.

Schema
------

**What is a Schema?**

A schema is a representation of a PyTree
structure consisting of a PyTree itself, wherein
the leaves are either None or a TensorValidator

When filled only with Nones, you are validating only the tensor
structure. Where a TensorValidator exists, that will be checked

We will also throw the result in a Schema wrapper class, just to be
safe.

**How is a Schema Used**

A downstream structure, perhaps a PyTreeValidator,
can consume a schema. It will verify the same or compatible
pytree exists, and apply tensor validators where they
exist.

**Schema Broadcasting**

If a schema is used to validate a tree where like nodes
in the schema terminate in a TensorValidator, but like nodes
in the tree terminate in a branch, *Schema Broadcasting* is
expected to occur.

Under this operation, for each subnode in the tree we
reuse the TensorValidator we saw earlier again and
again, applying it to each subnode.

**Specs**

Schema is a dataclass. It contains two fields:

- Schema: The PyTree representing the schema.

**Function: make_schema**

Make schema makes, as you might expect, a schema out of an
example pytree, and is designed to make schema use easier.

It behaves roughly like follows
def make_schema(pytree,
                construction_callback: Optional[Callable[[Any], Optional[TensorValidator] = None
                broadcast: bool = True,
                )->Schema:

    Walk the provided pytree, replacing leaves. When you hit a leaf:
    -  Start the result in state None
    -  If provided, call the construction callback with the leaf, and merge with current

    return the tree inside a Schema, with leaves replaced.



PyTreeValidator
---------------

A PyTreeValidator is responsible for consuming a schema and performing validation

**Main idea**

The PyTreeValidator can consume a schema at construction time, or recieve one
as a kwargs during runtime. Once it gets the schema, it will walk it in parallel
with the target operand tree and verify the schema and the tree match any TensorValidators


**Schema Tree Shape Validation**

There are several things we need to worry about to make sure the schema's are
the same.

Jax builtin utilities for handling trees includes a certain utility. This utility
is flatten_with_path. It will return a list of tuples containing the detected path
to a leaf, and then the leaf value itself, when you provide it with a pytree. By
providing a small helper function, it will also return something when the input
is None.

So long as we catch every leaf, including none, I think we can make it work.

*PyTree Structure Same*:

So long as the pytree structure is the same, this could prove it. We flatten both
trees. Then we walk through their shape/path tuples in parallel. So long as the
path keys are the same, which are in a conveniently hashable tuple, we accept the
structure to be the same.

While we are visiting the nodes, we can apply whatever other validation actions we
need

*PyTree Structures broadcastable*

In the case of a broadcast being required, it will eventually become
the case that the schema node_paths no longer are compatible with the
operand node_paths. Fortunately, this is also handlable. Two trees are
compatible for tree broadcast if they have the same shape until one of them
reaches a leaf

We check if the path for the operand is longer than the path for the
schema. If it is, it might be broadcastable. We then check if the path
of the operand, concatenated to the length of the schema path, are the same

If it is, we are in broadcast mode, and take appropriate action

*Pytrees incompatible*

Otherwise, the pytrees will be found to be incompatible.

** Specs **

fields:
- Common: A TensorValidator that is applied to all leaves.
- Mode: One of "header" or "tail". Lets the class know whether to
        apply common before anything leaf-specific, or after
- Schema: One of either a Schema, or a string specifying what the kwarg will be for it
- Broadcast: A bool. Whether to allow broadcast, or throw errors

methods

constructor:
    - The constructor is used to setup mode and schema information
    - The constructor accepts two parameters. One, required, is schema. This
      should either be a Schema or a string. The , Mode, is optional. It
      should be among "header" or "tail". Default will be header, for no particular
      reason.

    - Either way, we set the Schema and Mode

merge_tensor_validators:
    - Accepts a tensor validator
    - Merges it with Common, either as the head or tail depending on mode. Uses self

validate_leaf:
 - Validates whether a leaf is compatible
    - Accepts the leaf to validate, and the TensorValidator
    - Merges the TensorValidator with the common TensorValidator
    - Executes the validation against the leaf
    - return the result, which is either None or an Exception

validate:
- Validates whether a particular PyTree is compatible with a particular schema
-

The base schema specification

**Main idea**

This class will contain within it fields which can
represent the pytree.

This Pytree can have leaves of type TensorValidator or None.

Downstream, something will consume the schema and check that the
tree structure matches, and if the leaves are not none, apply the
TensorValidators to the given leaf positions.



PyTreeValidator
---------------

The basic PyTreeValidator

A PyTree Validator MUST get a schema from somewhere

**Schema**

The base schema class. It consists of fields, and some methods

fields:
- Structure: A PyTree. The leaves of the pytree Must


**BroadcastableSchema**

Records the proper structure a PyTree should posses. Any tensor structures
that are chained to this schema should automatically

**Identical


Round 1: Config Supported features
===================================

The user should have certain configuration options. I brainstorm, then
focus on the core pieces, here. See Decisions on Errors and Validation
for details.

considerations
--------------

Performance:
    - Validation inevitably steals some level of performance. Particularly if
      validation is going to involve python callbacks, it can have a severe
      negative effect
    - It would be great if we could shut it off as needed

Jit Compatibility:

-   One of the central issues here is that not all validation actions are
    jit-compatible. Getting around this means, in essence, either not performing
    validation based tensor values, or performing it in such a way that it cannot
    halt the execution of a program.
-   Ideas include allowing the user to specify when to raise errors, and printing
    issues to the console without

Logging:
    - It would be useful if the user can specify how to handle an error
      once it occurs.

Preliminary Config Options
--------------------------

Detail level.
    - Off: No validation
    - Basic: Only basic validation.
    - Advanced: Validation also inspects value-based criteria

- Error Converter: Mainly ensures jit statements are sane
    - Off: Raise directly as given
    - Warnings: Convert incompatible to console warnings
    - Checkify: Convert all to checkify statements.
    - Raise: Raise incompatible errors. Breaks jit compatibilty.

- Logging:
    - Console: Outputs sent to console, errors raises
    - Callback: Call a callback with the message and error
    - File: Print the message to a file

Round 2: Programmer wants
=========================

We consider what would be useful to have
when defining classes and associated validation actions
for the first time.

Setup once, use many times:
    - Ensure you can setup the validation mechanism for a process, then use it over and over again
    - This likely will happen before the class defintion.
    - Maybe a builder?
Compose existing validation:
    - It might be nice to, for instance, be able to compose a validator that says something is a probablity
      with one that checks for a dtype.
    - However, this might also end up looking confusing for the maintainer..
Flexible error types:
    - Despite this, I would still like to be able to raise an error of a particular type such
      as a value error or runtime error. At a minimum, I would like to include such
      information in my messages.
Hidden magic and extensibility
    - Configuration should be mostly done elsewhere from the location of
      the validation engine.
Relationships:
    - I should be able to setup schemas
    - I should be able to assert two features share a schema.
    - I should be able to generate a schema on the fly so I can assert pytrees match
Batch issues
    - I need to be able to elegantly handle validation involving batch shape.0
Messages:
    - Detailed error messages with arguments are needed to make best use of
      any validation mechanism
    - Hopefully, I should be able to provide an error message with formatting arguments,
      and provide the arguments itself at validation time.

Round 3: Preliminary Architecture
=================================

Propose the following preliminary architecture

Predicates Primitives Registry
------------------------------

- What is it?
    - The predicates primitives registry consists of string keys which indicate the name
      of a validation predicate such as "probability" or "floating", and code associated
      with that key which checks the satisfaction of the operand
    - Conceptually, it should be a define-once, use many object.
- How does it work?
    - There will exist a registry of validation predicates.
    - Each registry entry will consist of a name and a function returning a bool
    - There will be setters and support functions to set or get from the registry
- How to use it?

- There will exist a registry of validation predicates. Each predicate will consist of
  a name, which acts as a key, and a piece of code which accepts an operand to validate
  and returns a bool. False will indicate the operand failed validation
- The registry is intended to be populated in a central location
  such as a "validation" python file by validation primitives capable of
  checking things like dtype, probability satisfaction, and various other parameters
- The primitives are NOT associated with a particular situation yet, and do not
  have an error type tied to them.
- The primitives DO have tied to them postprocessing information such as whether
  or not they are jit compatible, and anything else I can think of that might
  be important.

Validator
---------

- What is it?
    - Validation occurs in objects known as "validators"
    - Conceptually, a "validator" is tied to a particular logical tensor or object.
    - It binds validation primitives to the local information needed to raise the proper
      error
    - It emits the proper error to the logging observer.
- How does it work?

Validator Builder
-----------------

Planned features
^^^^^^^^^^^^^^^^

Modes definition:

- Validation_Details_Level enum
    - Three modes
        - Off
        - Basics
        - Advanced: Default
    - Off: No validation occurs
    - Basics: Only things that do not break jit occur
    - Advance: All validation occurs

- Validation_Jit_Config
    - Debug: Cannot jit
    - Warnings: Can jit, raises warning to console
    - Checkify: Can jit, but must stage checkify. Raises errors in checkify.

- Logging_Mode enum:
    - Three modes:
        - Console: Raises issues to console
        - File: Dumps errors into a file along a particular path
        - Callback: Captures issue, pushes them into callback

Validation_Config dataclass:
    validation_detail_level: str =



Validation Operators:
- ValidationOperator:
    - Contains a single validation operation
    - Will have an validation predicate function. Must return true to pass
    - Will have a validation failed function. This will accept a formatted message
      and should promise to raise an error in some way.
    - Will have a raising
    - Will have a details message function. This is a format string
    - Will have a flag called
    - Has a __call__ mechanism that accepts
        - A operand mode
        - A validation mode
        -

- "Validator" class:

    - This is a combination of a builder and a callable.
      You might refer to it as a prototype? Anyhow,
      It is designed to be built once with the validation
      and warnings definitions provided, then executed many times.

      Unlike just about every other class, it is stateful. This is because,
      since it will be built once and only jitted after building, there will
      be nothing that can change between runs.

    - Builder mode -
        - Can attach error raising validation predicate function and issue details message
        - Can attach warning raising validation predicate function and issue details message
        - May define kwargs with message which to fill in with details message.
    - Callable mode - Will execute validation
            - Called with operand to validate,validation mode, and error message arguments
            - Executes validation.
            - Raises errors or prints warnings as appropriate.

Validation Registry:
    - A registry into which various validators can be placed.
    - Emphasizes the define-once, use-many nature of validators.
    - Register_validator:
        - Adds validator to the registry
        - Must define a name to get it from
    - Get_Validator:
        - Gets a validator that has been registered by name.

Support features
^^^^^^^^^^^^^^^^