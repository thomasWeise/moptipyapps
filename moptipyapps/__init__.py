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


def __setup() -> None:
    """Add `moptipyapps` to the dependencies unless called from pytest."""
    import inspect  # noqa # pylint: disable=C0415
    import moptipy.utils.sys_info  # noqa # pylint: disable=C0415

    for s in inspect.stack():
        if "pytest" in s.filename:
            return
    moptipy.utils.sys_info.add_dependency("moptipyapps")


__setup()
del __setup
