"""Run a small experiment applying RLS to one TSP instance."""

from moptipy.algorithms.so.rls import RLS
from moptipy.api.execution import Execution
from moptipy.operators.permutations.op0_shuffle import Op0Shuffle
from moptipy.operators.permutations.op1_swapn import Op1SwapN
from moptipy.spaces.permutations import Permutations

from moptipyapps.tsp.instance import Instance
from moptipyapps.tsp.tour_length import TourLength

# load the problem instance and define search space
instance = Instance.from_resource("gr17")  # pick instance gr17
space = Permutations.standard(instance.n_cities)

y = space.create()  # will later be used to hold the best solution found

# Build a single execution of a single run of a single algorithm, execute it,
# and store the best solution discovered in y and its length in `length`.
with Execution()\
        .set_rand_seed(1)\
        .set_solution_space(space)\
        .set_algorithm(  # This is the algorithm: Randomized Local Search.
            RLS(Op0Shuffle(space), Op1SwapN()))\
        .set_objective(TourLength(instance))\
        .set_max_fes(2048)\
        .execute() as process:
    process.get_copy_of_best_y(y)
    length = process.get_best_f()

print(f"tour length: {length}")
print(f"optimal length: {instance.tour_length_lower_bound}")
print(f"tour: {', '.join(str(i) for i in y)}")
