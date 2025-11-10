"""
Examples for differential equations systems with dynamic control.

Here we provide several systems whose state is governed by differential
equations that allow us to plug in a controller. The controller receives
the state of the system as input and its output is added to one (or more) of
the differential equations.

- The :mod:`~moptipyapps.dynamic_control.systems.stuart_landau` system has a
  two-dimensional state space. The goal is to navigate the system into the
  origin. Without control, the system will converge to oscillate in a circle
  around the origin with a diameter of sqrt(0.1).
- The :mod:`~moptipyapps.dynamic_control.systems.lorenz` system has a
  three-dimensional state space. The goal is again to move the system to the
  origin using a control strategy (and to keep it there). Without control, it
  will converge to a double-circle oscillation.
"""
