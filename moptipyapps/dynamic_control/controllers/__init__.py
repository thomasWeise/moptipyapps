"""
Several possible controllers for the dynamic control scenarios.

- :mod:`~moptipyapps.dynamic_control.controllers.linear` offers simple linear
  controllers.
- :mod:`~moptipyapps.dynamic_control.controllers.quadratic` offers simple
  quadratic controllers.
- :mod:`~moptipyapps.dynamic_control.controllers.cubic` offers simple
  cubic controllers.
- :mod:`~moptipyapps.dynamic_control.controllers.ann` offers poor man's
  artificial neural networks (ANNs), i.e., networks represented as plain
  functions whose parameter vectors are subject to optimization.
- :mod:`~moptipyapps.dynamic_control.controllers.predefined` provides a set of
  controllers taken from the works of NOACK, CORNEJO MACEDA, LI, and SUN.
- :mod:`~moptipyapps.dynamic_control.controllers.partially_linear` offers
  several linear controllers anchored at certain points in the state space and
  always uses the controller closest to the current state.
- :mod:`~moptipyapps.dynamic_control.controllers.min_ann` offers controllers
  similar to :mod:`~moptipyapps.dynamic_control.controllers.min_ann`, but
  instead of using the ANN output as controller value, these ANNs have an
  additional input `z` and then use the value `z*` as controller output for
  which the ANNs take on the smallest value.
- :mod:`~moptipyapps.dynamic_control.controllers.peaks` provides controllers
  similar to ANNs but using peak functions (`exp(-a²)`) as activation
  functions.

These controller blueprints can be used in two ways. The basic use case is to
synthesize, well, controllers for the dynamic system. In the dynamic system
controller synthesis problem, we have a system of differential equations
`D=ds/dt` where `D` depends on the current system state `s` and the output of
a controller `c`. So `D` is actually a function `D(s, c)`. Now the goal is to
get the dynamic system to move into a nice and stable state. We want to find
a controller `c` that can do this. Now `c` is actually a function `c(s, p)`
that takes as input the current system state `s` and its own parameterization
`p`. Imagine `c` to be, for example, an artificial neural network (an
:mod:`~moptipyapps.dynamic_control.controllers.ann`), then `p` is its weight
vector and `p` is subject to optimization. We would try to find the values `p`
that minimize a certain :mod:`~moptipyapps.dynamic_control.objective` function
that could, let's say, represent the cost or energy in the system. So use case
number one of our controllers is to represent the controller blueprints for
this process.

Use case number two is to use the controllers as blueprints for system models
`M`. Basically, a system model `M` should be a drop-in replacement for the
system equations `D`. In the *real world*, obviously, we do not have the
system equations that govern a complex system like a helicopter or something.
But we can test a helicopter controller in a wind tunnel. In such a scenario,
evaluating the objective function is very costly. If, instead, we could learn
a model `M` that can reasonably accurately describe the helicopter-experiment
behavior `D`, then we could replace the actual experiment with a simulation.
Since computing power is cheap, that would be awesome. Now it turns out that
the structure of the equations, i.e., a parameterized function relating inputs
to outputs, that we need for this, is pretty much the same as in use case
number one above. So we can re-use the same code, the basic
:mod:`~moptipyapps.dynamic_control.controller` blueprints, also for the
purpose of synthesizing models using a
:mod:`~moptipyapps.dynamic_control.model_objective` function.
"""
