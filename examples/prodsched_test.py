"""Show the data of a Production Scheduling Instance and Simulate it."""

from typing import Final

from moptipyapps.prodsched.instance import (
    Instance,
    to_stream,
)
from moptipyapps.prodsched.simulation import PrintingListener, Simulation

instance: Final[Instance] = Instance(
    name="test2", n_products=2, n_customers=1, n_machines=2, n_demands=2,
    routes=[[0, 1], [1, 0]],
    demands=[[0, 0, 1, 10, 20, 90], [1, 0, 0, 5, 22, 200]],
    warehous_at_t0=[2, 1],
    machine_product_unit_times=[[[10, 50, 15, 100], [5, 20, 7, 35, 4, 50]],
                                [[5, 24, 7, 80], [3, 21, 6, 50]]])

print("============ <INSTANCE DATA> ============")
for s in to_stream(instance):
    print(s)
print("============ </INSTANCE DATA> ============")

print()

print("============ <SIMULATION DATA> ============")
simulation: Final[Simulation] = Simulation(instance, PrintingListener())
simulation.ctrl_run()
print("============ </SIMULATION DATA> ============")
