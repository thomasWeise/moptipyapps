- [Applications](#3-applications)
  - [Dynamic Controller Synthesis](#33-dynamic-controller-synthesis)

### 3.3. Dynamic Controller Synthesis
Another interesting example for optimization is the synthesis of [active controllers for dynamic systems](https://thomasweise.github.io/moptipyapps/moptipyapps.dynamic_control.html).
Dynamic systems have a state that changes over time based on some laws.
These laws may be expressed as ordinary differential equations, for example.
The classical [Stuart-Landau system](https://thomasweise.github.io/moptipyapps/moptipyapps.dynamic_control.systems.html#module-moptipyapps.dynamic_control.systems.stuart_landau), for instance, represents an object whose coordinates on a two-dimensional plane change as follows:

```
sigma = 0.1 - x² - y²
dx/dt = sigma * x - y
dy/dt = sigma * y + x
```

Regardless on which `(x, y)` the object initially starts, it tends to move to a circular rotation path centered around the origin with radius `sqrt(0.1)`.
Now we try to create a controller `ctrl` for such a system that moves the object from this periodic circular path into a fixed and stable location.
The controller `ctrl` receives the current state, i.e., the object location, as input and can influence the system as follows:

```
sigma = 0.1 - x² - y²
c = ctrl(x, y)
dx/dt = sigma * x - y
dy/dt = sigma * y + x + c
```

What we try to find is the controller which can bring move object to the origin `(0, 0)` as quickly as possible while expending the least amount of force, i.e., having the smallest aggregated `c` values over time.
