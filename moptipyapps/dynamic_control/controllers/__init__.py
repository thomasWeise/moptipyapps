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
"""
