"""
Applications of Metaheuristic Optimization in Python.

Currently, the following applications are implemented:

- :mod:`~moptipyapps.binpacking2d` provides methods to solve two-dimensional
  bin packing instances,
- mod:`~moptipyapps.qap` offers instances of the well-known Quadratic
  Assignment Problem (QAP) and some very basic algorithms to tackle it,
- mod:`~moptipyapps.tsp` offers instances of the well-known Traveling
  Salesperson Problem (TSP) and some very basic algorithms to tackle it,
- mod:`~moptipyapps.ttp` offers instances of the Traveling Tournament Problem
  (TTP),.

The following additional tools are implemented:

- :mod:`~moptipyapps.tests` offers unit tests to try out optimization
  algorithms and other instances of :mod:`~moptipy.api.component` on the
  different problems that are provided above.
- :mod:`~moptipyapps.utils.shared` offers shared constants and tools.
"""

# During some local pip installs, the other already installed packages may not
# be accessible. This may cause
# `from moptipy.utils.sys_info import add_dependency` to fail during
# `pip3 --no-input --timeout 360 --retries 100 -v install .`.
# The reason is not clear to me.
# To prevent this issue, we now try to load the `moptipy` module and catch the
# corresponding `ModuleNotFoundError` exception. Only if the module is loaded
# correctly, we add `moptipyapps` to the dependencies.
# Of course, this ignores the situation where there would really be an error.
# But that is OK, because in that case we will crash later anyway.

can_do: bool = True
try:
    from moptipy.utils.sys_info import add_dependency
except ModuleNotFoundError:
    can_do = False

if can_do:
    add_dependency("moptipyapps", ignore_if_make_build=True)
