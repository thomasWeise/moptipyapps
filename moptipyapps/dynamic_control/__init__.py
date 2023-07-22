"""
Examples of dynamic control problems.

Here we have examples for dynamic control problems that are systems whose
state is governed by differential system which can be influenced by the
output of controllers.
Such systems can serve as primitive testbeds of actual dynamic control
scenarios such as flight control or other fluid dynamic systems. Different
from such more complicated systems, they are much faster and easier to
simulate, though. So we can play with them much more easily and quickly.

The initial starting point of the work here were conversations with
Prof. Dr. Bernd NOACK and Guy Yoslan CORNEJO MACEDA of the Harbin Institute
of Technology in Shenzhen, China (哈尔滨工业大学(深圳)) as well as the
following two MSc theses and book:

1. Yuxiang LI (李宇翔). Jet Mixing Enhancement using Deep Reinforcement
   Learning (基于深度强化学习的射流混合增强控制). MSc Thesis. Harbin Institute
   of Technology in Shenzhen, China (哈尔滨工业大学(深圳)).
   January 2023.
2. Wei SUN (孙伟). Wake Control of 1-2-3 Fluidic Pinball using Deep
   Reinforcement Learning (基于深度强化学习方法的 1-2-3 流体弹球尾流控制).
   MSc Thesis. Harbin Institute of Technology in Shenzhen, China
   (哈尔滨工业大学(深圳)). January 2023.
3. Guy Yoslan CORNEJO MACEDA, François LUSSEYRAN, and Bernd R. NOACK.
   xMLC: A Toolkit for Machine Learning Control, First Edition.
   Machine Learning Tools in Fluid Mechanics, Vol 2.
   Shenzhen & Paris; Universitätsbibliothek der Technischen Universität
   Braunschweig. 2022 https://doi.org/10.24355/dbbs.084-202208220937-0
"""
