# Potionomics Reinforcement Learning Environment

As stated in the description of this repository, this code is meant to simulate the mechanics involved in the potion crafting part of [PC game Potionomics](https://store.steampowered.com/app/1874490/Potionomics/).

## Motivation

The idea originated in an attempt to write a [Constraint Solver](https://en.wikipedia.org/wiki/Constraint_programming) that would find a set of optimal ingredient combinations to get a potion with certain properties. However, the project pivoted early on due to the realization that I needed to maintain some sort of internal state which, as far as I am aware, is not feasible in constraint programming.

The following reasons are why I chose to treat this part of the game as an RL problem.

### Deterministic State Transitions

Like many other games, potion crafting in Potionomics has deterministic state transitions. This means that, if you take an action in a given state, you will always transition to the next state. This means that the overall complexity of the problem is much lower than other games, and much simpler to write an environment for.

### Inherently Episodic

Since potion crafting has a definitive end (i.e., you've pressed the 'Craft' potion and are given a result), that means the environment is inherently episodic. For an agent whose only actions are to add ingredients to make a potion, that means that we can end the episode when one of the following things occurs:

- The agent chooses to end the episode (using a reserved action)
- The cauldron has reached its maximum item count
- The cauldron has reached its maximum magimin amount

This simplifies how to determine if the episode is over and also removes the need to have a 'timeout' to end or truncate the episode 'early'.

### Finite Action Space

The action space of this portion of Potionomics is finite and, relatively speaking, quite small. There are only 207 ingredients in the game, meaning the total action space is 208 (we reserve `0` for the agent to end the episode). Furthermore, unlike other environments (notably ones that involve physics), the action space is composed solely of discrete integers, where each integer maps to an ingredient.

### Well-understood State Space

The state space is well-understood in that the information about what is considered a 'good' potion is readily available (to a human player). Things like ratios, potion quality and potion traits, as well as how these interact with the end product, are accessible and easy to grasp.

However, converting the visual aids into computer-friendly terms was not as straightforward as explained later.

## Challenges

This section will detail challenges I faced in writing out this environment, which mostly revolved around having to reverse-engineer how the developers implemented their mechanics.

### Potion Price Calculation

This was one of the early challenges. Initially, I had made the assumption that potions were priced using some sort of function (i.e., [Geometric Sequence](https://en.wikipedia.org/wiki/Geometric_progression)). Had this been the case, then it would have been easy to implement the same system by performing some sort of polynomial fitting or linear regression on the data points.

However, the delta value between any two potion prices was not only inconsistent, different potions seemed to have different scaling as well. Some potions could be grouped together, but there did not seem to be any logical reason for the grouping either, leading me to believe it was more coincidental than planned behavior.

Ultimately, I solved this by using a [Lookup Table (LUT)](https://en.wikipedia.org/wiki/Lookup_table) and storing the information in the environment.

### Potion Quality Calculation

This was a similar challenge wherein I had mistakenly assumed a function of some sort would replicate the values, but ultimately I was unsuccessful in finding such an equation.

This problem was also solved by using a LUT and storing the information in the environment.

### How Traits Are Calculated

This mechanic was particularly tricky to figure out, as it is not an entirely deterministic process.

From testing multiple combinations of ingredients, I've determined that traits are calculated by the following:

1. Non-neutral traits on ingredients with lower magimin totals are overridden by non-neutral traits with higher magimin totals.
2. If two or more ingredients have the same magimin total, any overlapping traits are determined by uniform sampling. What remains unclear is whether multiple copies of a particular ingredient would influence the sampling or not.

### Defining a Reward Function

Human players can quickly grasp what ingredients work best, based on the potion and cauldron provided. Converting this knowledge into a reward function, however, was not as straightforward.
Currently, the reward function is defined as the following: $r=(1.0 - \Sigma\Delta_{ratios} * cauldron\_fullness) + Bonus_{stability}$

#### Cumulative Delta

The first component is the cumulative delta. This is the sum of the absolute distance between the expected magimin ratios of a potion and the magimin ratios of the contents of the cauldron. This is then masked to remove unrelated deltas before summation.

The logic is that, as the agent approaches the perfect recipe, the cumulative delta will approach zero.

<div align="center">
<table>
    <tr>
        <td colspan="2"><b>Definitions</b></td>
    </tr>
    <tr>
        <td>Magimins</td>
        <td>{A, B, C, D, E}</td>
    </tr>
    <tr>
        <td>Cauldron Ratio (C<sub>x</sub>)</td>
        <td>
            For each x in Magimins:<br>
            <code>C<sub>x</sub> = Total<sup>ingredients</sup><sub>x</sub> / Total<sup>cauldron</sup><sub>x</sub></code>
        </td>
    </tr>
    <tr>
        <td>Recipe Ratio (R<sub>x</sub>)</td>
        <td>
            For each x in Magimins:<br>
            <code>R<sub>x</sub> = Recipe<sub>x</sub> / Σ Recipe</code>
        </td>
    </tr>
    <tr>
        <td>Cauldron Ratios</td>
        <td>
            <code>&lt;C<sub>A</sub>, C<sub>B</sub>, C<sub>C</sub>, C<sub>D</sub>, C<sub>E</sub>&gt;</code>
        </td>
    </tr>
    <tr>
        <td>Recipe Ratios</td>
        <td>
            <code>&lt;R<sub>A</sub>, R<sub>B</sub>, R<sub>C</sub>, R<sub>D</sub>, R<sub>E</sub>&gt;</code>
        </td>
    </tr>
    <tr>
        <td>Mask</td>
        <td>
            <code>ratios<sup>mask</sup><sub>i</sub> = 1</code> if <code>ratios<sup>recipe</sup><sub>i</sub> ≠ 0</code>, else 0
        </td>
    </tr>
    <tr>
        <td>Cumulative Delta</td>
        <td>
            <code>Σ Δ<sub>ratios</sub> = Σ (|ratios<sub>cauldron</sub> - ratios<sub>recipe</sub>| × ratios<sup>mask</sup>)</code>
        </td>
    </tr>
</table>
</div>

#### Stability Bonus

In-game, potions that have higher stability have a higher chance to gain one or more stars, which directly corresponds to the price a potion can be sold at. Therefore, players have an incentive to create high-stability potions.

However, to avoid adding more complexity to the environment, I opted to turn this into a bonus reward instead. This also allowed me to narrow the task down to "whether the agent can make a functional potion or not".

The stability bonus was originally calculated using this equation, which I came up with arbitrarily.

One final note, the co-efficient $1.661$ was chosen because the bonus reward becomes $1$ with perfect ratios.
<div align="center">
    <p>
        <b>&epsilon; = 1 &times; 10<sup>-8</sup></b>
    </p>
    <p>
        <b>Bonus<sub>stability</sub> = 1.661 &times; log<sub>10</sub>(stability<sub>potion</sub> + &epsilon;)</b>
    </p>
</div>

However, the complexity of the equation did not really prove worthwhile, nor did having such a large negative reward for the `CANNOTMAKE` tier. Therefore, I instead opted to use an arbitrary sequence of values:

<div align="center">
    <p>
        <b>Bonus<sub>stability</sub> = 0.25 &times; x</b>
    </p>
</div>

This formula is straightforward, yet it still provides meaningful rewards for producing high-stability potions.

#### Percent of Cauldron Used

Raising potion quality relies heavily on filling up the cauldron's magimin capacity as much as possible. Therefore, the agent should be given incentive to do the same, while still focusing on creating functional potions. This equation: $cauldron\_fullness=\Sigma(Magimins_{ingredients})/Cauldron_{MagiminCapcity}$ is multiplied against the cumulative delta to try and encourage the agent to balance both aspects equally.

#### Potion Price

While this is not included in the agent's reward function, the potion price does play an obvious role in how 'good' the outcome is. It also serves as the main point of contention for the contests in the game. However, because the player usually creates more than one potion per craft, transforming the profit into something more meaningful to the agent is not as simple as it seems.

My current thought is using something akin to a sigmoid function to normalize the expected potion profit to the [0, 1] range, but this doesn't work very well when the expected profit can be orders of magnitude higher than the cost of the ingredients. Additionally, there are times where making a potion at a loss is recommended or necessary due to other factors (e.g., custom orders, contests, etc.), so negative rewards aren't exactly ideal either.
