"""
A small template for ROP-based experiments.

This experiment uses the NSGA-II algorithm to optimize Re-Order-Points (ROPs)
to achieve both a high worst-case fillrate and a low worst-case average stock
level.

It is just a preliminary idea.

In manufacturing systems, the concept of re-order points (ROPs) can be used
to schedule production.
The ROP basically provides one limit `X` per product.
If there are less or equal (`<=`) `X` units of the product in the warehouse,
a production order is issued so that one new unit is produced.

The question is how should `X` be set so that we can

1. satisfy as many customer demands as possible directly when they come in,
   i.e., maximize the fill rate
   (:attr:`~moptipyapps.prodsched.statistics.Statistics.immediate_rates`,
   represented by
   :mod:`~moptipyapps.prodsched.objectives.worst_and_mean_fill_rate`) and
2. have a low total average number of product units sitting around in our
   warehouse, i.e., minimize the stock level
   (:attr:`~moptipyapps.prodsched.statistics.Statistics.stock_levels`,
   represented by
   :mod:`~moptipyapps.prodsched.objectives.max_stocklevel`).

Since we have `n=10` products in the Thürer [1] base scenario, we also have
`10` such ROP values `X`.

Now how can we know the fill rate and the stock level?
This is done by simulating the whole system in action.
We therefore use instances generated using the Thürer-style distributions as
defined in :mod:`~moptipyapps.prodsched.mfc_generator` [1].

Our simulations (see :mod:`~moptipyapps.prodsched.simulation`) are not
randomized.
Instead, each of them is based on a fixed
:mod:`~moptipyapps.prodsched.instance`.
Each :mod:`~moptipyapps.prodsched.instance.Instance` defines exactly when a
customer :class:`~moptipyapps.prodsched.instance.Demand` comes in and,
for many time windows, the production time needed by a machine to produce
one unit of product.
Of course, all of these values follow the random distributions (Erlang, Gamma,
with respective parameters) given in Tables 2 and 3 of the original Thürer
paper [1] and implemented in :mod:`~moptipyapps.prodsched.mfc_generator`.
But apart from this, they are fixed per instance.

This means that we can take the same ROP and run the simulation twice for a
given instance and will get exactly the same results and
:mod:`~moptipyapps.prodsched.statistics` (fill rates, stock levels, etc.).

Of course, using a single fixed :mod:`~moptipyapps.prodsched.instance` may
be misleading.
Maybe we would think that a certain ROP is very good ... but it is only
good on the specific instance we tried.

So as a second step, we here generate 11 instances (via
:func:`~moptipyapps.prodsched.instances.get_instances`).
And then we look at the worst fill rate and the worst stock level over all 11
instances (more or less).
And we use that to judge whether an ROP is good or not.

This leaves the question:
Where do these ROPs come from?

They come from an optimization process.
First, we define that ROPs be integer vectors (i.e., from an
:class:`~moptipy.spaces.intspace.IntSpace`) where each element comes from
the range `0..63`.
We sample the initial solutions randomly from that interval
(via :class:`~moptipy.operators.intspace.op0_random.Op0Random`).

As unary search operator, we take an existing ROP and, for a number of
elements, sample a new value normally distributed around it (with standard
deviation 2.5) but rounded to integer.
We do this for a binomially distributed number of elements, exactly like the
bit-string based (1+1) EA would do it, but implemented for integers by
operator :class:`~moptipy.operators.intspace.op1_mnormal.Op1MNormal`.

As binary operator, we use uniform crossover, given as
:class:`~moptipy.operators.intspace.op2_uniform.Op2Uniform`.
This operator takes two existing solutions and creates a new solution by
element-wise copying elements of the parents.
For each element, it randomly decides from which parent solution it should be
copied.

As optimization algorithm, we use NSGA-II implemented by class
:class:`~moptipy.algorithms.mo.nsga2.NSGA2`.
This is a multi-objective optimization algorithm.
We do this because we have two goals:

1. Maximize the worst-case fill rates,
2. Minimize the worst-case average stock level.

Regarding the first objective, we have a small tweak:
Assume that, over all 11 instances, `PM` be the worst fill rate per product
(in [0,1], 0 being worst) and `AM` be the worst average fill rate over all
products (0 worst, 1 best).
Then our objective value -- subject to minimization -- is
`(1 - PM) * 100 + (1 - AM)`.
This is implemented in module
:mod:`~moptipyapps.prodsched.objectives.worst_and_mean_fill_rate`.

The objective function minimizing the stock level is implemented in module
:mod:`~moptipyapps.prodsched.objectives.max_stocklevel`.

In summary, what we do is this:

1. The optimization algorithm proposes ROPs, each of which being an integer
   vector with 10 values (1 value per product).
   (:class:`~moptipy.spaces.intspace.IntSpace`)
   These integer vectors are the elements of the search space.

2. The ROP is evaluated by simulating it 11 times (using 10000 time units per
   instance, 3000 of which are used for warmup).
   This is implemented as an :mod:`~moptipy.api.encoding`,
   :class:`~moptipyapps.prodsched.rop_multisimulation.ROPMultiSimulation`.

3. This :mod:`~moptipyapps.prodsched.simulation`-based decoding procedure maps
   the ROP vectors to the solution space. In our case, this solution space are
   just multi-statistics records, as implemented in module
   :mod:`~moptipyapps.prodsched.multistatistics`, where a corresponding
   :mod:`~moptipy.api.space` implementation is also provided.

4. Each of the 11 simulations is based on one fixed
   :mod:`~moptipyapps.prodsched.instance`, where all customer demands and
   machine work times (at certain time intervals) are fixed (but were sampled
   based on the distributions given in the Thürer paper [1]) using module
   :mod:`~moptipyapps.prodsched.mfc_generator`.

5. Each of the 11 simulations has per-product fill rates `PM_i,j`, one average
   fill rate `AM_i`, one average stock level `SL_i` stored in the
   :mod:`~moptipyapps.prodsched.multistatistics` records.

6. As first objective, we use the smallest `PM_i_j` as `PM` and the smallest
   `AM_i` as `AM` and compute `(1 - AM) * 100 + (1 - PM)` in
   :mod:`~moptipyapps.prodsched.objectives.worst_and_mean_fill_rate`.

7. As second objective, we use the largest `SL_i` in
   :mod:`~moptipyapps.prodsched.objectives.max_stocklevel`.

8. NSGA-II, given in :mod:`~moptipy.algorithms.mo.nsga2`, maintains a
   population of solutions which it evaluates like that.

9. NSGA-II decides which solutions to keep based on the current Pareto front
   and crowding distance.

10. The retained solutions are reproduced, either via unary or binary search
    operators.

11. The unary search operator changes a random number (binomial distribution)
    of elements of a ROP vector (via normal distribution).
    It is given in
    :class:`~moptipy.operators.intspace.op1_mnormal.Op1MNormal`.

12. The binary search operator is simple uniform crossover, i.e., fills a
    new solution with elements of either of the two parent solutions. It is
    defined in :class:`~moptipy.operators.intspace.op2_uniform.Op2Uniform`.

The core idea is that we do not use randomized ARENA-like simulation.
Instead, we use a simulator that is based on deterministic, fully pre-defined
instances.
These instances are still randomly generated according to the distributions
given by Thürer in [1].
However, they have all values pre-determined.
This allows us to run a simulation with the same ROP twice and get the exactly
same result.
If two different ROPs are simulated, but both of them decide to produce 1 unit
of product `A` on machine `V` at time unit `T`, then for both of them this
will take exactly the same amount of time.

Simulation results are thus less noisy.

Of course, there is a danger of overfitting.
This is why we need to use multiple simulations to check a ROP.
And then we take the worst-case results.

So this is the idea.

1. Matthias Thürer, Nuno O. Fernandes, Hermann Lödding, and Mark Stevenson.
   Material Flow Control in Make-to-Stock Production Systems: An Assessment of
   Order Generation, Order Release and Production Authorization by Simulation
   Flexible Services and Manufacturing Journal. 37(1):1-37. March 2025.
   doi: https://doi.org/10.1007/s10696-024-09532-2
"""


import argparse
from typing import Final

from moptipy.algorithms.mo.nsga2 import NSGA2
from moptipy.api.experiment import run_experiment
from moptipy.api.mo_execution import MOExecution
from moptipy.mo.problem.weighted_sum import WeightedSum
from moptipy.operators.intspace.op0_random import Op0Random
from moptipy.operators.intspace.op1_mnormal import Op1MNormal
from moptipy.operators.intspace.op2_uniform import Op2Uniform
from moptipy.spaces.intspace import IntSpace
from pycommons.io.console import logger
from pycommons.io.path import Path
from pycommons.types import check_int_range

from moptipyapps.prodsched.instance import Instance
from moptipyapps.prodsched.instances import get_instances
from moptipyapps.prodsched.multistatistics import MultiStatisticsSpace
from moptipyapps.prodsched.objectives.max_stocklevel import MaxStockLevel
from moptipyapps.prodsched.objectives.worst_and_mean_fill_rate import (
    WorstAndMeanFillRate,
)
from moptipyapps.prodsched.rop_multisimulation import ROPMultiSimulation
from moptipyapps.utils.shared import moptipyapps_argparser


def run(dest: str, instances: str, n_inst: int, n_runs: int,
        max_fes: int, ps: int) -> None:
    """
    Run the experiment.

    :param dest: the destination directory
    :param instances: the directory with the instances
    :param n_inst: the number of instances
    :param n_runs: the number of runs
    :param max_fes: the maximum FEs
    :param ps: the population size
    """
    logger(f"Beginning experiment with dest={dest!r}, instances={instances!r}"
           f", n_inst={n_inst}, n_runs={n_runs}, and max_fes={max_fes}.")
    use_dest: Final[Path] = Path(dest)
    use_dest.ensure_dir_exists()
    logger(f"Destination folder is {use_dest!r}.")

    use_insts: Final[Path] = Path(instances)
    use_insts.ensure_dir_exists()
    logger(f"Instances folder is {use_insts!r}.")

    check_int_range(n_inst, "n_inst", 1, 128)
    check_int_range(max_fes, "max_fes", 10, 10 ** 10)
    check_int_range(ps, "ps", 4, 16384)

    logger(f"Loading {n_inst} instances from {use_insts!r}.")
    insts: Final[tuple[Instance, ...]] = get_instances(n_inst, instances)
    if tuple.__len__(insts) != n_inst:
        raise ValueError("Could not load required instances.")
    logger(f"Loaded {n_inst} instances from {use_insts!r}.")

    n_prod: int | None = None
    for inst in insts:
        if n_prod is None:
            n_prod = inst.n_products
        elif n_prod != inst.n_products:
            raise ValueError("Inconsistent number of products!")
    if n_prod is None:
        raise ValueError("No instances?")

    search_space: Final[IntSpace] = IntSpace(n_prod, 0, 63)
    op0: Final[Op0Random] = Op0Random(search_space)
    op1: Final[Op1MNormal] = Op1MNormal(search_space, sd=2.5)
    op2: Final = Op2Uniform()
    algo: Final[NSGA2] = NSGA2(op0, op1, op2, ps, 1 / min(16, ps))
    encoding: Final[ROPMultiSimulation] = ROPMultiSimulation(insts)
    f1: Final[WorstAndMeanFillRate] = WorstAndMeanFillRate()
    f2: Final[MaxStockLevel] = MaxStockLevel()
    ws: Final[WeightedSum] = WeightedSum((f1, f2), (
        (1 / (f1.upper_bound() - f1.lower_bound())), 1 / (2 * n_prod)))
    solution_space: Final[MultiStatisticsSpace] = MultiStatisticsSpace(insts)

    def __setup(_) -> MOExecution:
        """
        Set up the experiment.

        :return: the execution
        """
        return (MOExecution()
                .set_search_space(search_space)
                .set_algorithm(algo)
                .set_solution_space(solution_space)
                .set_objective(ws)
                .set_encoding(encoding)
                .set_max_fes(max_fes)
                .set_log_improvements(True))

    run_experiment(base_dir=use_dest, instances=(lambda: "all", ),
                   setups=(__setup, ), n_runs=n_runs,
                   pre_warmup_fes=2, perform_warmup=False,
                   perform_pre_warmup=True)


# Run the experiment from the command line
if __name__ == "__main__":
    parser: Final[argparse.ArgumentParser] = moptipyapps_argparser(
        __file__, "ROP-based MFC Optimization",
        "Run a small experiment with ROP-based MFC optimization.")
    parser.add_argument(
        "dest", help="the directory to store the experimental results under",
        type=Path, nargs="?", default="./results/")
    parser.add_argument(
        "insts", help="the directory with the instances",
        type=Path, nargs="?", default="./instances/")
    parser.add_argument(
        "n_inst", help="the number of instances",
        type=int, nargs="?", default=11)
    parser.add_argument(
        "n_runs", help="the number of runs",
        type=int, nargs="?", default=31)
    parser.add_argument(
        "max_fes", help="the number of FEs per run",
        type=int, nargs="?", default=8192)
    parser.add_argument(
        "ps", help="the population size of NSGA-II",
        type=int, nargs="?", default=64)
    args: Final[argparse.Namespace] = parser.parse_args()
    run(args.dest, args.insts, args.n_inst, args.n_runs, args.max_fes,
        args.ps)
