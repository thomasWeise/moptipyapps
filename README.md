[![make build](https://github.com/thomasWeise/moptipyapps/actions/workflows/build.yaml/badge.svg)](https://github.com/thomasWeise/moptipy/actions/workflows/build.yaml)
[![pypi version](https://img.shields.io/pypi/v/moptipyapps)](https://pypi.org/project/moptipyapps)
[![pypi downloads](https://img.shields.io/pypi/dw/moptipyapps.svg)](https://pypistats.org/packages/moptipyapps)
[![coverage report](https://thomasweise.github.io/moptipyapps/tc/badge.svg)](https://thomasweise.github.io/moptipyapps/tc/index.html)


# moptipyapps: Applications of Metaheuristic Optimization in Python

- [Introduction](#1-introduction)
- [Installation](#2-installation)
- [Applications](#3-applications)
  - [Two-Dimensional Bin Packing Problem](#31-two-dimensional-bin-packing-problem)
  - [Traveling Salesperson Problem (TSP)](#32-the-traveling-salesperson-problem-tsp)
- [Unit Tests and Static Analysis](#4-unit-tests-and-static-analysis)
- [License](#5-license)
- [Contact](#6-contact)


## 1. Introduction

[`moptipy`](https://thomasweise.github.io/moptipy/) is a library with implementations of metaheuristic optimization methods in Python&nbsp;3.10 that also offers an environment for replicable experiments ([`flyer`](https://thomasweise.github.io/moptipy/_static/moptipy_flyer.pdf)).
[`moptipyapps`](https://thomasweise.github.io/moptipyapps) is a collection of applications and experiments based on `moptipy`.


## 2. Installation

In order to use this package and to, e.g., run the example codes, you need to first install it using [`pip`](https://pypi.org/project/pip/) or some other tool that can install packages from [PyPi](https://pypi.org).
You can install the newest version of this library from [PyPi](https://pypi.org/project/moptipyapps/) using [`pip`](https://pypi.org/project/pip/) by doing

```shell
pip install moptipyapps
```

This will install the latest official release of our package as well as [all dependencies](https://thomasweise.github.io/moptipyapps/requirements.html).
If you want to install the latest source code version from GitHub (which may not yet be officially released), you can do

```shell
pip install git+https://github.com/thomasWeise/moptipyapps.git
```

If you want to install the latest source code version from GitHub (which may not yet be officially released) and you have set up a private/public key for GitHub, you can also do:

```shell
git clone ssh://git@github.com/thomasWeise/moptipyapps
pip install moptipyapps
```

This may sometimes work better if you are having trouble reaching GitHub via `https` or `http`.

You can also clone the repository and then run a [`make` build](https://thomasweise.github.io/moptipyapps/Makefile.html), which will automatically install all dependencies, run all the tests, and then install the package on your system, too.
This will work only on Linux, though.
It also installs the [dependencies for building](https://thomasweise.github.io/moptipyapps/requirements-dev.html), which include, e.g., those for [unit testing and static analysis](#4-unit-tests-and-static-analysis).
If this build completes successful, you can be sure that [`moptipyapps`](https://thomasweise.github.io/moptipyapps) will work properly on your machine.

All dependencies for using and running `moptipyapps` are listed at [here](https://thomasweise.github.io/moptipyapps/requirements.html).
The additional dependencies for a [full `make` build](https://thomasweise.github.io/moptipyapps/Makefile.html), including unit tests, static analysis, and the generation of documentation are listed [here](https://thomasweise.github.io/moptipyapps/requirements-dev.html).


## 3. Applications

Here we list the applications of [`moptipy`](https://thomasweise.github.io/moptipy).


### 3.1. Two-Dimensional Bin Packing Problem

In the package [`moptipyapps.binpacking2d`](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#module-moptipyapps.binpacking2d), we provide tools for experimenting and playing around with the two-dimensional bin packing problem.
Bin packing is a classical domain from Operations Research.
The goal is to pack objects into containers, the so-called bins.
We address [two-dimensional rectangular bin packing](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#module-moptipyapps.binpacking2d).
We provide the bin packing [instances](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#module-moptipyapps.binpacking2d.instance) from [2DPackLib](https://site.unibo.it/operations-research/en/research/2dpacklib) as [resources](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#moptipyapps.binpacking2d.instance.Instance.from_resource) together with [this package](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#module-moptipyapps.binpacking2d).
Each such instances defines a set of `n_different_items` objects `Oi` with `i` from `1..n_different_objects`.
Each object `Oi` is a rectangle with a given width and height.
The object occur is a given multiplicity `repetitions(O_i)`, i.e., either only once or multiple times.
The bins are rectangles with a given width and height too.
The goal of tackling such an instance is to package all the objects into as few as possible bins.
The objects therefore may be rotated by 90 degrees.

We address this problem by representing a packing as a [signed permutation with repetitions](https://thomasweise.github.io/moptipy/moptipy.spaces.html#module-moptipy.spaces.signed_permutations) of the numbers `1..n_different_objects`, where the number `i` occurs `repetitions(O_i)` times.
If an object is to be placed in a rotated way, this is denoted by using `-i` instead of `i`.
Such permutations are processed from beginning to end, placing the objects into bins as they come according to some heuristic encoding.
We provide two variants of the Improved Bottom Left encoding.
[The first one](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#module-moptipyapps.binpacking2d.ibl_encoding_1) closes bins as soon as one object cannot be placed into them.
[The second one](https://thomasweise.github.io/moptipyapps/moptipyapps.binpacking2d.html#module-moptipyapps.binpacking2d.ibl_encoding_2) tries to put each object in the earliest possible bin.
While the former one is faster, the latter one leads to better packings.

We can then apply a black-box metaheuristic to search in the space of these signed permutations with repetitions.
The objective function would be some measure consistent with the goal of minimizing the number of bins used.

*Examples:*

- [plot a packing chart](https://thomasweise.github.io/moptipyapps/examples/binpacking2d_plot.html)
- [apply a randomized local search to one 2D bin packing instance](https://thomasweise.github.io/moptipyapps/examples/binpacking2d_rls.html)
- [measure the runtime of the different encodings for the 2D bin packing problem](https://thomasweise.github.io/moptipyapps/examples/binpacking2d_timing.html)

Important work on this code has been contributed by Mr. Rui ZHAO (赵睿), <zr1329142665@163.com>, a Master's student at the Institute of Applied Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School of Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei University (合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of Prof. Dr. Thomas Weise (汤卫思教授), who then refined the implementations.


### 3.2. The Traveling Salesperson Problem (TSP)

In the package [`moptipyapps.tsp`](https://thomasweise.github.io/moptipyapps/moptipyapps.tsp.html#module-moptipyapps.tsp), we provide tools to run experiments and play around with the Traveling Salesperson Problem (TSP) .
A TSP instance is defined as a fully-connected graph with `n_cities` nodes.
Each edge in  the graph has a weight, which identifies the distance between the nodes.
The goal is to find the *shortest* tour that visits every single node in the graph exactly once and then returns back to its starting node.
Then nodes are usually called cities.
A tour can be represented in path representation, which means that it is stored as a permutation of the numbers `0` to `n_cities-1`.
The number at index `k` identifies that `k`-th city to visit.
So the first number in the permutation identifies the first city, the second number the second city,
and so on.
The length of the tour can be computed by summing up the distances from the `k`-th city to the `k+1`-st city, for `k` in `0..n_cities-2` and then adding the distance from the last city to the first city.

We use the TSP instances from [TSPLib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), the maybe most important benchmark set for the TSP.
110 of these instances are included as resources in this package.

*Examples:*

- [apply a randomized local search to one TSP instance](https://thomasweise.github.io/moptipyapps/examples/tsp_rls.html)

Important work on this code has been contributed by Mr. Tianyu LIANG (梁天宇), <liangty@stu.hfuu.edu.cn> a Master's student at the Institute of Applied Optimization (应用优化研究所, http://iao.hfuu.edu.cn) of the School of Artificial Intelligence and Big Data (人工智能与大数据学院) at Hefei  University (合肥学院) in Hefei, Anhui, China (中国安徽省合肥市) under the supervision of Prof. Dr. Thomas Weise (汤卫思教授).


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


## 4. Unit Tests and Static Analysis

When developing and applying randomized algorithms, proper testing and checking of the source code is of utmost importance.
If we apply a randomized metaheuristic to an optimization problem, then we usually do not which solution quality we can achieve.
Therefore, we can usually not know whether we have implemented the algorithm correctly.
In other words, detecting bugs is very hard.
Unfortunately, this holds also for the components of the algorithms, such as the search operators, especially if they are randomized as well.
A bug may lead to worse results and we might not even notice that the worse result quality is caused by the bug.
We may think that the algorithm is just not working well on the problem.

Therefore, we need to test all components of the algorithm as far as we can.
We can try check, for example, if a randomized nullary search operator indeed creates different solutions when invoked several times.
We can try to check whether an algorithm fails with an exception.
We can try to check whether the search operators create valid solutions and whether the algorithm passes valid solutions to the objective function.
We can try to whether an objective function produces finite objective values and if bounds are specified for the objective values, we can check whether they indeed fall within these bounds.
Now we cannot prove that there are no such bugs, due to the randomization.
But by testing a few hundred times, we can at least detect very obvious and pathological bugs.

To ease such testing for you, we provide a set of tools for testing implemented algorithms, spaces, and operators in the package [moptipyapps.tests](https://thomasweise.github.io/moptipyapps/moptipyapps.tests.html).
Here, you can find functions where you pass in instances of your implemented components and they are checked for compliance with the [moptipy API](https://thomasweise.github.io/moptipy/moptipy.api.html) and the problem setups defined in `moptipyapps`.
In other words, if you go and implement your own algorithms, operators, and optimization problems, you can use our pre-defined unit tests to give them a thorough check before using them in production.
Again, such tests cannot prove the absence of bugs.
But they can at least give you a fair shot to detect pathological errors before wasting serious experimentation time.

We also try to extensively test our own code, see the coverage report of [`moptipy`](https://thomasweise.github.io/moptipy/tc/index.html) and [`moptipyapps`](https://thomasweise.github.io/moptipyapps/tc/index.html).

Another way to try to improve and maintain code quality is to use static code analysis and type hints where possible and reasonable.
A static analysis tool can inform you about, e.g., unused variables, which often result from a coding error.
It can tell you if the types of expressions do not match, which usually indicates a coding error, too.
It can tell you if you perform some security-wise unsafe operations (which is less often a problem in optimization, but it does not hurt to check). 
Code analysis tools can also help you to enforce best practices, which are good for performance, readability, and maintainability.
They can push you to properly format and document your code, which, too, improve readability, maintainability, and usability.
They even can detect a set of well-known and frequently-occurring bugs.
We therefore also run a variety of such tools on our code base, including (in alphabetical order):

- [`autoflake`](https://pypi.org/project/autoflake/), a tool for finding unused imports and variables
- [`bandit`](https://pypi.org/project/bandit/), a linter for finding security issues
- [`dodgy`](https://pypi.org/project/dodgy/), for checking for dodgy looking values in the code
- [`flake8`](https://pypi.org/project/flake8/), a collection of linters
- [`flake8-bugbear`](http://pypi.org/project/flake8-bugbear), for finding common bugs
- [`flake8-eradicate`](http://pypi.org/project/flake8-eradicate), for finding commented-out code
- [`flake8-use-fstring`](http://pypi.org/project/flake8-use-fstring), for checking the correct use of f-strings
- [`mypy`](https://pypi.org/project/mypy/), for checking types and type annotations
- [`pycodestyle`](https://pypi.org/project/pycodestyle/), for checking the formatting and coding style of the source
- [`pydocstyle`](https://pypi.org/project/pydocstyle/), for checking the format of the docstrings
- [`pyflakes`](https://pypi.org/project/pyflakes/), for detecting some errors in the code
- [`pylint`](https://pypi.org/project/pylint/), another static analysis tool
- [`pyroma`](https://pypi.org/project/pyroma/), for checking whether the code complies with various best practices
- [`ruff`](https://pypi.org/project/ruff/), a static analysis tool checking a wide range of coding conventions
- [`semgrep`](https://pypi.org/project/semgrep/), another static analyzer for finding bugs and problems
- [`tryceratops`](https://pypi.org/project/tryceratops/), for checking against exception handling anti-patterns
- [`unimport`](https://pypi.org/project/unimport/), for checking against unused import statements
- [`vulture`](https://pypi.org/project/vulture/), for finding dead code

On git pushes, GitHub also automatically runs [CodeQL](https://codeql.github.com/) to check for common vulnerabilities and coding errors.
We also turned on GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/repository-security-advisories/configuring-private-vulnerability-reporting-for-a-repository) and the Dependabot [vulnerability](https://docs.github.com/en/code-security/dependabot/dependabot-alerts/configuring-dependabot-alerts) and [security](https://docs.github.com/en/code-security/dependabot/dependabot-security-updates/configuring-dependabot-security-updates) alerts.

Using all of these tools increases the build time.
However, combined with thorough unit testing and documentation, it should help to prevent bugs, to improve readability, maintainability, and usability of the code.
It does not matter whether we are doing research or try to solve practical problems in the industry &mdash; we should always strive to make good software with high code quality.

Often, researchers in particular think that hacking something together that works is enough, that documentation is unimportant, that code style best practices can be ignored, and so on.
And then they wonder why they cannot understand their own code a few years down the line (at least, this happened to me in the past&hellip;).
Or why no one can use their code to build atop of their research (which is the normal case for me).

Improving code quality can *never* come later.
We *always* must maintain high coding and documentation standards from the very beginning.
While `moptipy` may still be far from achieving these goals, at least we try to get there.

Anyway, you can find our [full `make` build](https://thomasweise.github.io/moptipyapps/Makefile.html) running all the tests, doing all the static analyses, creating the documentation, and creating and packaging the distribution files [here](https://thomasweise.github.io/moptipyapps/Makefile.html).
Besides the [basic `moptipyapps` dependencies](https://thomasweise.github.io/moptipyapps/requirements-dev.html), it requires [a set of additional dependencies](https://thomasweise.github.io/moptipyapps/requirements-dev.html).
These are all automatically installed during the build procedure.
The build only works under Linux.


## 5. License

[`moptipyapps`](https://thomasweise.github.io/moptipyapps) is a library for implementing, using, and experimenting with metaheuristic optimization algorithms.
Our project is developed for scientific, educational, and industrial applications.

Copyright (C) 2023  [Thomas Weise](http://iao.hfuu.edu.cn/5) (汤卫思教授)

Dr. Thomas Weise (see [Contact](#6-contact)) holds the copyright of this package *except* for the data of the benchmark sets we imported from other sources.
`moptipyapps` is provided to the public as open source software under the [GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007](https://thomasweise.github.io/moptipyapps/LICENSE.html).
Terms for other licenses, e.g., for specific industrial applications, can be negotiated with Dr. Thomas Weise (who can be reached via the [contact information](#6-contact) below).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.

Please visit the [contributions guidelines](https://thomasweise.github.io/moptipyapps/CONTRIBUTING.html) for `moptipy` if you would like to contribute to our package.
If you have any concerns regarding security, please visit our [security policy](https://thomasweise.github.io/moptipyapps/SECURITY.html).


### 5.1. Exceptions

- The included benchmark instance data of the [two-dimensional bin packing](#31-two-dimensional-bin-packing-problem) is taken from [2DPackLib](https://site.unibo.it/operations-research/en/research/2dpacklib).
  It has been stored in a more size-efficient way and some unnecessary information has been stripped from it (as we really only need the raw bin packing data).
  Nevertheless, the copyright of the original data lies with the authors [2DPackLib](https://site.unibo.it/operations-research/en/research/2dpacklib) or the original authors of the datasets used by them.
- The included benchmark instances for the [Traveling Salesperson Problem](#32-the-traveling-salesperson-problem-tsp) are taken from [TSPLib](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).
  The copyright of the original data lies with Gerhard Reinelt, the original author of TSPLib, or the original authors of the datasets used by him.


## 6. Contact

If you have any questions or suggestions, please contact
Prof. Dr. [Thomas Weise](http://iao.hfuu.edu.cn/5) (汤卫思教授) of the 
Institute of Applied Optimization (应用优化研究所, [IAO](http://iao.hfuu.edu.cn)) of the
School of Artificial Intelligence and Big Data ([人工智能与大数据学院](http://www.hfuu.edu.cn/aibd/)) at
[Hefei University](http://www.hfuu.edu.cn/english/) ([合肥学院](http://www.hfuu.edu.cn/)) in
Hefei, Anhui, China (中国安徽省合肥市) via
email to [tweise@hfuu.edu.cn](mailto:tweise@hfuu.edu.cn) with CC to [tweise@ustc.edu.cn](mailto:tweise@ustc.edu.cn).
