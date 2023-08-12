# Potionomics Reinforcement Learning Environment

As stated in the description of this repository, this code is meant to simulate the mechanics involved in the potion crafting part of [PC game Potionomics](https://store.steampowered.com/app/1874490/Potionomics/).

## Motivation

The idea originated in an attempt to write a [Constraint Solver](https://en.wikipedia.org/wiki/Constraint_programming) that would find a set of optimal ingredient combinations to get a potion with certain properties. However, the project pivoted early on due to the realization that I needed to maintain some sort of internal state which, as far as I am aware, is not feasible in constraint programming.

The following reasons are why I chose to treat this part of the game as an RL problem.

### Deterministic State Transitions

Like many other games, potion crafting in Potionomics has deterministic state transitions. This means that, if you take an action in a given state, you will always transition to the next state. This means that the overall complexity of the problem is much lower than other games, and much simpler to write an environment for.

### Inherently Episodic

Since potion crafting has a definitive end (i.e., you've pressed the 'Craft' potion and are given a result), that means the environment is inherently episodic. For an agent whose only actions are to add ingredients to make a potion, that means that we can end the episode when the cauldron is full or the agent chooses to no longer add ingredients.

This simplifies how to determine if the episode is over and also removes the need to have a 'timeout' to end or truncate the episode early.

### Finite Action Space

The action space of this portion of Potionomics is finite and, relatively speaking, quite small. There are only 207 ingredients in the game, meaning the total action space is 208 (we reserve 0 for our own purposes). Furthermore, unlike other environments (notably ones that involve physics), the action space is composed solely of discrete integers, where each integer maps to an ingredient.

### Well-understood State Space

The state space is well-understood in that the information about what is considered a 'good' potion is readily available (to a human player). Things like ratios, potion quality and potion traits, as well as how these interact with the end product, are accessible and easy to grasp.

However, converting the visual aids into computer-friendly terms was not as straightforward as explained later.

## Challenges

This section will detail challenges I faced in writing out this environment, which mostly revolved around having to reverse-engineer how the developers implemented their mechanics.

### Potion Price Calculation

This was one of the early challenges. Initially, I had made the assumption that potions were priced using some sort of function (i.e., Geometric Sequence). Had this been the case, then it would have been easy to implement the same system by performing some sort of polynomial fitting or linear regression on the data points.

However, the delta value between any two potion prices was not only inconsistent, different potions seemed to have different scaling as well. Some potions could be grouped together, but there did not seem to be any logical reason for the grouping either, leading me to believe it was more coincidental than planned behavior.

Ultimately, I solved this by using a Lookup Table (LUT) and storing the information in the environment.

### Potion Quality Calculation

This was a similar challenge wherein I had mistakenly assumed a function of some sort would replicate the values, but ultimately I was unsuccessful in finding such an equation.

This problem was also solved by using a LUT and storing the information in the environment.

### How Traits Are Calculated

This mechanic was particularly tricky to figure out, as it is not an entirely deterministic process.

From testing multiple combinations of ingredients, I've determined that traits are calculated by the following:

1. Non-neutral traits on ingredients with lower magimin totals are overridden by non-neutral traits with higher magimin totals.
2. If two or more ingredients have the same magimin total, any overlapping traits are determined by uniform sampling. What remains unclear is whether multiple copies of a particular ingredient would influence the sampling or not.
