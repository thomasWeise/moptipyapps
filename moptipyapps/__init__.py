"""
Applications of Metaheuristic Optimization in Python.

Currently, the following applications are implemented:

- :mod:`~moptipyapps.binpacking2d` provides methods to solve two-dimensional
  bin packing instances.

The following additional tools are implemented:

- :mod:`~moptipyapps.tests` offers unit tests to try out optimization
  algorithms and other instances of :mod:`~moptipy.api.component` on the
  different problems that are provided above.
- :mod:`~moptipyapps.shared` offers shared constants and tools.
"""

from moptipy.utils.sys_info import add_dependency

add_dependency("moptipyapps", ignore_if_make_build=True)
