"""
A class to model a dynamic system governed by differential equations.

A system has a current state vector at any point in time. The state changes
over time based on differential equations. These equations can be influenced
by the output of a :mod:`~moptipyapps.dynamic_control.controller`. Our
:class:`System` presents the state dimension and differential equations. It
also presents several starting states for simulating the system. The starting
states are divided into training and testing states. The training states can
be used for, well, training controllers to learn how to handle the system.
The testing states are then used to verify whether a synthesized controller
actually works.

Examples for different dynamic systems are given in package
:mod:`~moptipyapps.dynamic_control.systems`.
"""

from typing import Any, Callable, Final, Iterable, TypeVar

import moptipy.utils.plot_utils as pu
import numpy as np
from moptipy.api.component import Component
from moptipy.utils.logger import KeyValueLogSection
from moptipy.utils.nputils import array_to_str
from moptipy.utils.path import Path
from moptipy.utils.types import check_to_int_range, type_error

from moptipyapps.dynamic_control.ode import multi_run_ode
from moptipyapps.dynamic_control.results_log import ResultsLog
from moptipyapps.dynamic_control.results_plot import ResultsPlot

#: the color for training points
_TRAINING_COLOR: Final[str] = "orange"
#: the color for testing points
_TEST_COLOR: Final[str] = "cyan"

#: the type variable for ODE controller parameters
T = TypeVar("T")


def _dummy_ctrl(_: np.ndarray, __: float, ___: Any,
                out: np.ndarray) -> None:
    """Do nothing as internal dummy controller that does nothing."""
    out.fill(0.0)


class System(Component):
    """A class for governing a system via differential system."""

    def __init__(self, name: str, state_dims: int, control_dims: int,
                 test_starting_states: np.ndarray,
                 training_starting_states: np.ndarray,
                 test_steps: int = 5000,
                 training_steps: int = 1000,
                 plot_examples: Iterable[int] = (0, )) -> None:
        """
        Initialize the system.

        :param name: the name of the system.
        :param state_dims: the state dimensions
        :param control_dims: the control dimensions
        :param test_starting_states: the starting states to be used for
            testing, as a matrix with one point per row
        :param test_steps: the steps to be taken for the test simulation
        :param training_steps: the steps to be taken for the training
            simulation
        :param training_starting_states: the starting states to be used for
            training, as a matrix with one point per row
        :param plot_examples: the points that should be plotted
        """
        super().__init__()
        if not isinstance(name, str):
            raise type_error(name, "name", str)
        #: the name of the model
        self.name: Final[str] = name
        #: the dimensions of the state variable
        self.state_dims: Final[int] = check_to_int_range(
            state_dims, "state_dims", 2, 3)
        #: the dimensions of the controller output
        self.control_dims: Final[int] = check_to_int_range(
            control_dims, "control_dims", 1, 100)

        if not isinstance(test_starting_states, np.ndarray):
            raise type_error(test_starting_states, "test_starting_states",
                             np.ndarray)
        if (len(test_starting_states) < 1) or (
                test_starting_states.shape[1] != state_dims):
            raise ValueError(
                f"invalid test starting states {test_starting_states!r}.")
        #: the test starting states
        self.test_starting_states: Final[np.ndarray] = test_starting_states

        if not isinstance(training_starting_states, np.ndarray):
            raise type_error(training_starting_states,
                             "training_starting_states", np.ndarray)
        if (len(training_starting_states) < 1) or (
                training_starting_states.shape[1] != state_dims):
            raise ValueError("invalid training starting "
                             f"states {training_starting_states!r}.")
        #: the test starting states
        self.training_starting_states: Final[np.ndarray] = \
            training_starting_states

        #: the test simulation steps
        self.test_steps: Final[int] = check_to_int_range(
            test_steps, "test_steps", 10, 1_000_000_000)
        #: the training simulation steps
        self.training_steps: Final[int] = check_to_int_range(
            training_steps, "training_steps", 10, 1_000_000_000)

        #: the plot examples
        self.plot_examples: Final[tuple[int, ...]] = tuple(sorted(set(
            plot_examples)))
        total: Final[int] = len(self.training_starting_states) \
            + len(self.test_starting_states) - 1
        for i in self.plot_examples:
            check_to_int_range(i, "plot_examples[i]", 0, total)

    def log_parameters_to(self, logger: KeyValueLogSection) -> None:
        """
        Log all parameters of this component as key-value pairs.

        :param logger: the logger for the parameters
        """
        super().log_parameters_to(logger)
        logger.key_value("stateDims", self.state_dims)
        logger.key_value("controlDims", self.control_dims)
        logger.key_value("testStartingStates",
                         array_to_str(self.test_starting_states.flatten()))
        logger.key_value(
            "trainingStartingStates",
            array_to_str(self.training_starting_states.flatten()))
        logger.key_value("testSteps", self.test_steps)
        logger.key_value("trainingSteps", self.training_steps)
        logger.key_value("examplePlots", array_to_str(np.array(
            self.plot_examples, int).flatten()))

    def __str__(self):
        """
        Get the name of this model.

        :return: the name of this model
        """
        return self.name

    def equations(self, state: np.ndarray, time: float,
                  control: np.ndarray, out: np.ndarray) -> None:
        """
        Compute the values of the differential equations to be simulated.

        :param state: the state of the system
        :param time: the time index
        :param control: the output of the controller
        :param out: the differential, i.e., the output of this function
        """

    def plot_points(self, dest_dir: str, skip_if_exists: bool = True) -> Path:
        """
        Plot the training and testing points of the equation.

        :param dest_dir: the destination directory
        :param skip_if_exists: `True` to skip existing files, `False` otherwise
        :return: the plotted file
        """
        name: Final[str] = f"{self.name}_points"
        dest_folder: Final[Path] = Path.path(dest_dir)
        dest_folder.ensure_dir_exists()
        dest_file: Final[Path] = dest_folder.resolve_inside(f"{name}.pdf")
        if dest_file.ensure_file_exists() and skip_if_exists:
            return dest_file

        figure = pu.create_figure()
        if self.state_dims == 3:
            axes = figure.add_subplot(projection="3d")
            axes.set_aspect("equal")
            axes.force_zorder = True
            axes.set_zlabel("z")

            axes.scatter(self.training_starting_states[:, 0],
                         self.training_starting_states[:, 1],
                         self.training_starting_states[:, 2],
                         color=_TRAINING_COLOR, marker="x",
                         zorder=1)
            axes.scatter(self.test_starting_states[:, 0],
                         self.test_starting_states[:, 1],
                         self.test_starting_states[:, 2],
                         color=_TEST_COLOR, marker="o", zorder=2)
            axes.scatter(0, 0, 0, color="black", marker="+", zorder=3)
            zlim = axes.get_zlim()
            axes.set_zlim(1.1 * zlim[0], 1.1 * zlim[1])
        elif self.state_dims == 2:
            axes = pu.get_axes(figure)
            axes.set_aspect("equal")
            axes.grid(True)
            axes.scatter(self.training_starting_states[:, 0],
                         self.training_starting_states[:, 1],
                         color=_TRAINING_COLOR, marker="x",
                         zorder=1)
            axes.scatter(self.test_starting_states[:, 0],
                         self.test_starting_states[:, 1],
                         color=_TEST_COLOR, marker="o", zorder=2)
        else:
            raise ValueError(f"invalid dimensions {self.state_dims} "
                             f"for {self}.")

        xlim = axes.get_xlim()
        axes.set_xlim(1.1 * xlim[0], 1.1 * xlim[1])
        ylim = axes.get_ylim()
        axes.set_ylim(1.1 * ylim[0], 1.1 * ylim[1])
        axes.set_title(f"{self.name} system\ntraining points ("
                       f"{_TRAINING_COLOR}), test points ({_TEST_COLOR})")

        axes.set_xlabel("x")
        axes.set_ylabel("y")
        res = pu.save_figure(figure, name, dest_folder, "pdf")[0]
        if res != dest_file:
            raise ValueError(
                f"Wrong destination file {res!r}!={dest_file!r}?")
        return dest_file

    def describe_system(
            self, title: str | None,
            controller: Callable[[np.ndarray, float, T, np.ndarray], None],
            parameters: T, base_name: str, dest_dir: str,
            skip_if_exists: bool = False) \
            -> tuple[Path, Path]:
        """
        Describe the performance of a given system of system.

        :param title: the title
        :param controller: the controller to simulate
        :param parameters: the controller parameters
        :param base_name: the base name of the file to produce
        :param dest_dir: the destination directory
        :param skip_if_exists: if the file already exists
        :returns: the paths of the generated files
        """
        dest: Final[Path] = Path.path(dest_dir)
        dest.ensure_dir_exists()

        the_title = f"{self.name} system"
        if title is not None:
            the_title = f"{the_title}: {title}"

        file_1 = dest.resolve_inside(f"{base_name}_plot.pdf")
        file_2 = dest.resolve_inside(f"{base_name}_results.csv")
        if file_1.ensure_file_exists() and file_2.ensure_file_exists() \
                and skip_if_exists:
            return file_1, file_2

        with ResultsPlot(dest_file=file_1, sup_title=the_title,
                         state_dims=self.state_dims,
                         plot_indices=self.plot_examples) as plt, \
                ResultsLog(self.state_dims, file_2) as log:
            multi_run_ode(
                self.test_starting_states,
                self.training_starting_states,
                (plt.collector, log.collector),
                self.equations,
                controller, parameters, self.control_dims,
                self.test_steps, self.training_steps)

        return file_1, file_2

    def describe_system_without_control(self, dest_dir: str,
                                        skip_if_exists: bool = True) \
            -> tuple[Path, Path]:
        """
        Describe the performance of a given system of system.

        :param dest_dir: the destination directory
        :param skip_if_exists: if the file already exists
        :returns: the paths of the generated files
        """
        return self.describe_system("without control", _dummy_ctrl, 0.0,
                                    f"{self.name}_without_control", dest_dir,
                                    skip_if_exists)
