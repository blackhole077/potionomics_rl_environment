import csv
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import logging
import numpy as np
import torch

from potionomics_env.potionomics_enum import PotionomicsPotionStability
from potionomics_env.potionomics_enum import PotionomicsPotionTier
from potionomics_env.schemas import (
    PotionomicsCauldron,
    PotionomicsIngredient,
    PotionomicsPotionRecipe,
)

# This is a placeholder value to ensure that all observations are the correct length. If you choose to encode the state information to ensure fixed-length then this would no longer be necessary.
POTIONOMICS_MAX_ITEMS_IN_CAULDRON = 14
logger = logging.getLogger(__name__)


class PotionomicsEnvironment(gym.Env):
    """A custom Gym environment that is meant to simulate the potion-crafting mechanic present in Potionomics.

    Attributes:

        all_ingredients List[PotionomicsIngredient]: A list of all ingredients available in the game. Used to create the action space for the agent.

        all_recipes List[PotionomicsPotionRecipe]: A list of all recipes available in the game. For each episode, a recipe is sampled uniformly from this list and given as part of the state observation.

        all_cauldrons List[PotionomicsCauldron]: A list of all cauldrons available in the game. For each episode, a cauldron is sampled uniformly from this list and given as part of the state observation.

        cauldron PotionomicsCauldron: The cauldron that the agent uses for a given episode.

        recipe PotionomicsPotionRecipe: The potion recipe that the agent uses for a given episode.

        current_ingredients List[PotionomicsIngredient]: A list of ingredients that the agent has placed into the cauldron at a given timestep. It is given to the agent as part of the state observation.

        current_stability PotionomicsPotionStability: An enum that indicates how stable the potion is. Used as part of the state observation and reward function.

        potion_tier: PotionomicsPotionTier: An enum that indicates the quality of the potion wherein higher numbers are better. Used as part of the state observation and reward function.

        potion_magimin_thresholds_array np.ndarray: A Lookup Table (LUT) that informs how high quality a potion is. Used to determine potion price.

        current_base_price int: The current price of the potion, calculated based on its total magimin content. Used as part of the state observation and reward function.

        cost_of_items int: The cost to make the potion assuming all ingredients were purchased. Used as part of the state observation and reward function.

        potion_prices_dict Dict[str, np.ndarray]: A mapping that maps potions to a Lookup Table (LUT) of prices, as Potionomics does not appear to use a function to calculate price. Used in calculating the price of a potion.

        action_space gym.spaces.Discrete: The action space of the agent within this environment.

        _action_to_ingredient Dict[int, PotionomicsIngredient]: A mapping from integer to ingredient in Potionomics.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        all_ingredients: List[PotionomicsIngredient],
        all_recipes: List[PotionomicsPotionRecipe],
        all_cauldrons: List[PotionomicsCauldron],
    ):
        self.all_ingredients = all_ingredients
        self.all_recipes = all_recipes
        self.all_cauldrons = all_cauldrons
        # BUG: Basically since each cauldron has different amounts of things you can put in, the size of the observation wouldn't be the same. So for now we're not dealing with that since we'd need to encode the state.
        # self.cauldron: PotionomicsCauldron = self.all_cauldrons[
        #     np.random.choice(len(self.all_cauldrons), 1)[0]
        # ].model_copy(deep=True)
        self.cauldron: PotionomicsCauldron = self.all_cauldrons[-1]
        self.cauldron.setup()
        # self.recipe: PotionomicsPotionRecipe = self.all_recipes[
        #     np.random.choice(len(self.all_recipes), 1)[0]
        # ]
        self.recipe: PotionomicsPotionRecipe = self.all_recipes[4]
        self.num_ingredients: np.ndarray = np.ascontiguousarray(
            np.random.randint(
                low=1,
                high=self.cauldron.max_num_items_allowed + 1,
                size=len(self.all_ingredients) + 1,
            )
        )
        self.num_ingredients[0] = 0  # Set the None action to have 0
        self.current_ingredients: List[int] = []
        self.current_stability = PotionomicsPotionStability.CANNOTMAKE
        self.potion_tier: PotionomicsPotionTier = None
        self.potion_traits: np.ndarray = np.zeros((5,))
        self.potion_magimin_thresholds_array = np.array(
            [
                [0, 10, 20, 30, 40, 50],
                [60, 75, 90, 105, 115, 130],
                [150, 170, 195, 215, 235, 260],
                [290, 315, 345, 370, 400, 430],
                [470, 505, 545, 580, 620, 660],
                [720, 800, 875, 960, 1040, 1125],
            ],
            order="C",
        )
        self.current_base_price: int = 0
        self.cost_of_items: int = 0
        self.potion_prices_dict: Dict[str, np.ndarray] = {
            "Health_Potion": np.array(
                [
                    [16, 18, 20, 23, 26, 29],
                    [36, 41, 46, 51, 58, 65],
                    [81, 91, 103, 116, 130, 146],
                    [183, 206, 231, 260, 293, 330],
                    [412, 463, 521, 587, 660, 742],
                    [928, 1044, 1174, 1321, 1486, 1672],
                ],
                order="C",
            ),
            "Mana_Potion": np.array(
                [
                    [20, 23, 25, 28, 32, 36],
                    [45, 51, 57, 64, 72, 81],
                    [101, 114, 128, 144, 163, 183],
                    [229, 257, 289, 325, 366, 412],
                    [515, 579, 652, 733, 825, 928],
                    [1160, 1305, 1468, 1651, 1858, 2090],
                ],
                order="C",
            ),
            "Stamina_Potion": np.array(
                [
                    [22, 25, 28, 31, 35, 40],
                    [50, 56, 63, 71, 79, 89],
                    [112, 126, 141, 159, 179, 201],
                    [251, 283, 318, 358, 403, 453],
                    [566, 637, 717, 806, 907, 1021],
                    [1276, 1435, 1615, 1817, 2044, 2299],
                ],
                order="C",
            ),
            "Speed_Potion": np.array(
                [
                    [24, 27, 30, 34, 38, 43],
                    [54, 61, 68, 77, 87, 97],
                    [122, 137, 154, 173, 195, 219],
                    [274, 309, 347, 391, 439, 494],
                    [618, 695, 782, 880, 990, 1113],
                    [1392, 1566, 1761, 1982, 2229, 2508],
                ],
                order="C",
            ),
            "Tolerance_Potion": np.array(
                [
                    [28, 32, 35, 40, 45, 50],
                    [63, 71, 80, 90, 101, 114],
                    [142, 160, 180, 202, 228, 256],
                    [320, 360, 405, 456, 513, 577],
                    [721, 811, 912, 1026, 1155, 1299],
                    [1624, 1827, 2055, 2312, 2601, 2926],
                ],
                order="C",
            ),
            "Fire_Tonic": np.array(
                [
                    [18, 20, 23, 26, 29, 32],
                    [41, 46, 51, 58, 65, 73],
                    [91, 103, 116, 228, 146, 165],
                    [206, 231, 260, 293, 330, 371],
                    [463, 521, 587, 660, 742, 835],
                    [1044, 1174, 1321, 1486, 1672, 1881],
                ],
                order="C",
            ),
            "Ice_Tonic": np.array(
                [
                    [20, 23, 25, 28, 32, 36],
                    [45, 51, 57, 64, 72, 81],
                    [101, 114, 128, 144, 163, 183],
                    [229, 257, 289, 325, 366, 412],
                    [515, 579, 652, 733, 825, 928],
                    [1160, 1305, 1468, 1651, 1858, 2090],
                ],
                order="C",
            ),
            "Thunder_Tonic": np.array(
                [
                    [22, 25, 28, 31, 35, 40],
                    [50, 56, 63, 71, 79, 89],
                    [112, 126, 141, 159, 179, 201],
                    [251, 283, 318, 358, 403, 453],
                    [566, 637, 717, 806, 907, 1021],
                    [1276, 1435, 1615, 1817, 2044, 2299],
                ],
                order="C",
            ),
            "Shadow_Tonic": np.array(
                [
                    [24, 27, 30, 34, 38, 43],
                    [54, 61, 68, 77, 87, 97],
                    [122, 137, 154, 173, 195, 219],
                    [274, 309, 347, 391, 439, 494],
                    [618, 695, 782, 880, 990, 1113],
                    [1392, 1566, 1761, 1982, 2229, 2508],
                ],
                order="C",
            ),
            "Radiation_Tonic": np.array(
                [
                    [26, 29, 33, 37, 42, 47],
                    [59, 66, 74, 83, 94, 106],
                    [132, 148, 167, 188, 211, 238],
                    [297, 334, 376, 423, 476, 535],
                    [669, 753, 847, 953, 1072, 1206],
                    [1508, 1696, 1908, 2147, 2417, 2700],
                ],
                order="C",
            ),
            "Sight_Enhancer": np.array(
                [
                    [20, 23, 25, 28, 32, 36],
                    [45, 51, 57, 64, 72, 81],
                    [101, 114, 128, 144, 163, 183],
                    [229, 257, 289, 325, 366, 412],
                    [515, 579, 652, 733, 825, 928],
                    [1160, 1305, 1468, 1651, 1858, 2090],
                ],
                order="C",
            ),
            "Alertness_Enhancer": np.array(
                [
                    [26, 29, 33, 37, 42, 47],
                    [59, 66, 74, 83, 94, 106],
                    [132, 148, 167, 188, 211, 238],
                    [297, 334, 376, 423, 476, 535],
                    [669, 753, 847, 953, 1072, 1206],
                    [1508, 1696, 1908, 2147, 2415, 2717],
                ],
                order="C",
            ),
            "Insight_Enhancer": np.array(
                [
                    [24, 27, 30, 34, 38, 43],
                    [54, 61, 68, 77, 87, 97],
                    [122, 137, 154, 173, 195, 219],
                    [274, 309, 347, 391, 439, 494],
                    [618, 695, 782, 880, 990, 1113],
                    [1392, 1566, 1761, 1982, 2229, 2508],
                ],
                order="C",
            ),
            "Dowsing_Enhancer": np.array(
                [
                    [28, 32, 35, 40, 45, 50],
                    [63, 71, 80, 90, 101, 114],
                    [142, 160, 180, 202, 228, 256],
                    [320, 360, 405, 456, 513, 577],
                    [721, 811, 912, 1026, 1155, 1299],
                    [1624, 1827, 2055, 2312, 2601, 2926],
                ],
                order="C",
            ),
            "Seeking_Enhancer": np.array(
                [
                    [32, 36, 41, 46, 51, 58],
                    [72, 81, 91, 103, 115, 130],
                    [162, 183, 205, 231, 260, 293],
                    [366, 411, 463, 521, 586, 659],
                    [824, 927, 1043, 1173, 1320, 1485],
                    [1856, 2088, 2349, 2642, 2973, 3344],
                ],
                order="C",
            ),
            "Poison_Cure": np.array(
                [
                    [19, 21, 24, 27, 30, 34],
                    [43, 48, 54, 61, 69, 77],
                    [96, 108, 122, 137, 154, 174],
                    [217, 244, 275, 309, 348, 391],
                    [489, 550, 619, 696, 784, 881],
                    [1102, 1240, 1395, 1569, 1765, 1986],
                ],
                order="C",
            ),
            "Drowsiness_Cure": np.array(
                [
                    [21, 24, 27, 30, 34, 38],
                    [47, 53, 60, 67, 76, 85],
                    [107, 120, 135, 152, 171, 192],
                    [240, 270, 304, 342, 384, 433],
                    [541, 608, 684, 770, 866, 974],
                    [1218, 1370, 1541, 1734, 1951, 2195],
                ],
                order="C",
            ),
            "Petrification_Cure": np.array(
                [
                    [22, 25, 28, 31, 35, 40],
                    [50, 56, 63, 71, 79, 89],
                    [112, 126, 141, 159, 179, 201],
                    [251, 283, 318, 358, 403, 453],
                    [566, 637, 717, 806, 907, 1021],
                    [1276, 1435, 1615, 1817, 2044, 2299],
                ],
                order="C",
            ),
            "Silence_Cure": np.array(
                [
                    [20, 23, 25, 28, 32, 36],
                    [45, 51, 57, 64, 72, 81],
                    [101, 114, 128, 144, 163, 183],
                    [229, 257, 289, 325, 366, 412],
                    [515, 579, 652, 733, 825, 928],
                    [1160, 1305, 1468, 1651, 1858, 2090],
                ],
                order="C",
            ),
            "Curse_Cure": np.array(
                [
                    [25, 28, 32, 36, 40, 45],
                    [56, 63, 71, 80, 90, 101],
                    [127, 143, 161, 181, 203, 229],
                    [286, 321, 362, 407, 458, 515],
                    [645, 725, 815, 915, 1030, 1160],
                    [1450, 1630, 1835, 2065, 2320, 2650],
                ],
                order="C",
            ),
        }
        ### GYM-SPECIFIC INTIALIZATIONS ###
        self.action_space = gym.spaces.Discrete(len(all_ingredients) + 1)
        self._action_to_ingredient: Dict[int, PotionomicsIngredient] = dict(
            zip(np.arange(1, len(all_ingredients) + 1), self.all_ingredients)
        )
        self._action_to_ingredient[0] = None

    def insert_item(self, index_of_item_to_insert: int) -> int:
        """Attempt to insert an item into the cauldron.

        Three conditions must be satisfied for the item to be inserted into the cauldron:
            1. The item must not be None
            2. There must be space in the cauldron for the item
            3. The item must not exceed the remaining magimins available

        :param index_of_item_to_insert: The integer corresponding to the item to insert into the cauldron.
        :type index_of_item_to_insert: int
        :return: An integer indicating whether the item was inserted into the cauldron successfully. Used in calculating the reward.
        :rtype: int
        """

        item_to_insert: PotionomicsIngredient = self._action_to_ingredient[
            index_of_item_to_insert
        ]
        # logger.info(f"Attempting to Insert {item_to_insert.name}\tStock: {self.num_ingredients[index_of_item_to_insert]}")
        has_enough_of_item: bool = self.num_ingredients[index_of_item_to_insert] > 0
        magimin_check = (
            self.cauldron.current_total_magimin_amount
            + item_to_insert.total_magimin_value
            <= self.cauldron.max_total_magimin_allowed
        )
        if (
            len(self.current_ingredients) < self.cauldron.max_num_items_allowed
            and has_enough_of_item
            and magimin_check
            and item_to_insert is not None
        ):
            self.cauldron.current_a_magimin_amount += item_to_insert.a_magimin_value
            self.cauldron.current_b_magimin_amount += item_to_insert.b_magimin_value
            self.cauldron.current_c_magimin_amount += item_to_insert.c_magimin_value
            self.cauldron.current_d_magimin_amount += item_to_insert.d_magimin_value
            self.cauldron.current_e_magimin_amount += item_to_insert.e_magimin_value
            self.cauldron.current_total_magimin_amount += (
                item_to_insert.total_magimin_value
            )
            self.cauldron.current_num_items += 1
            self.current_ingredients.append(index_of_item_to_insert)
            self.cost_of_items += item_to_insert.item_price
            self.num_ingredients[index_of_item_to_insert] -= 1
            # if self.num_ingredients[index_of_item_to_insert] < 1:
            #     logger.warning(f"{item_to_insert.name} has no more stock.")
            return 1
        else:
            #             logger.debug(
            #                 f"""
            # Not Legal Because:
            # Enough Space?: {len(self.current_ingredients) < self.cauldron.max_num_items_allowed}
            # Enough of Item?: {has_enough_of_item}
            # Passes Magimin Check?: {magimin_check}
            # Item to Insert?: {item_to_insert}
            # """
            #             )
            return 0

    def calculate_potion_rank_and_price(self) -> None:
        """Calculate the rank and price of the potion.

        Since Potionomics does not seem to use a function for
        calculating potion rank nor price, a Lookup Table (LUT)
        is used instead.
        """

        # When going from 2D to 1D arrays, we tend to use this equation for indexing: y = (row*num_cols) + col
        # This function is going from 1D to 2D arrays. Thus we first find y (i.e., potion_index) and then solve for row and col.
        potion_index: int = np.searchsorted(
            self.potion_magimin_thresholds_array.flatten(),
            self.cauldron.current_total_magimin_amount,
        )
        if potion_index >= self.potion_magimin_thresholds_array.size:
            potion_index -= 1  # Need to make sure we stay in bounds

        self.potion_tier = PotionomicsPotionTier(
            potion_index // self.potion_magimin_thresholds_array.shape[0]
        )
        row = potion_index // self.potion_magimin_thresholds_array.shape[0]
        col = potion_index - (row * self.potion_magimin_thresholds_array.shape[0])
        self.current_base_price = self.potion_prices_dict.get(self.recipe.name)[row][
            col
        ]

    def calculate_current_magimin_ratios(self) -> np.ndarray:
        """Calculate the normalized magimin content of all ingredients in the cauldron.

        :return: An array of normalized magimin values with shape (5, ) where each element corresponds to a category of magimin.
        :rtype: np.ndarray
        """

        current_a_magimin_ratio: float = (
            (
                self.cauldron.current_a_magimin_amount
                / self.cauldron.current_total_magimin_amount
            )
            if self.cauldron.current_total_magimin_amount > 0
            else 0
        )
        current_b_magimin_ratio: float = (
            (
                self.cauldron.current_b_magimin_amount
                / self.cauldron.current_total_magimin_amount
            )
            if self.cauldron.current_total_magimin_amount > 0
            else 0
        )
        current_c_magimin_ratio: float = (
            (
                self.cauldron.current_c_magimin_amount
                / self.cauldron.current_total_magimin_amount
            )
            if self.cauldron.current_total_magimin_amount > 0
            else 0
        )
        current_d_magimin_ratio: float = (
            (
                self.cauldron.current_d_magimin_amount
                / self.cauldron.current_total_magimin_amount
            )
            if self.cauldron.current_total_magimin_amount > 0
            else 0
        )
        current_e_magimin_ratio: float = (
            (
                self.cauldron.current_e_magimin_amount
                / self.cauldron.current_total_magimin_amount
            )
            if self.cauldron.current_total_magimin_amount > 0
            else 0
        )
        return np.array(
            [
                current_a_magimin_ratio,
                current_b_magimin_ratio,
                current_c_magimin_ratio,
                current_d_magimin_ratio,
                current_e_magimin_ratio,
            ]
        )

    def calculate_current_stability(self) -> float:
        """Calculate the current stability of the potion.

        Potion stability in Potionomics is determined by the cumulative delta(s)
        between each type of magimin present in the recipe. Afterwards, there are
        certain threshold values (found empirically) that determine the stability.

        :return: A floating-point value that indicates how much the current potion is off from the 'perfect' ratios.
        :rtype: float
        """

        current_magimin_ratios = self.calculate_current_magimin_ratios()
        recipe_magimin_ratios: np.ndarray = self.recipe.magimin_ratios
        magimin_deltas: np.ndarray = np.abs(
            recipe_magimin_ratios - current_magimin_ratios
        )
        cumulative_delta: float = (
            (self.recipe.magimin_ratios_int > 0).astype(np.int_) * magimin_deltas
        ).sum()
        if cumulative_delta >= 0.25:
            self.current_stability = PotionomicsPotionStability.CANNOTMAKE
        elif cumulative_delta >= 0.15:
            self.current_stability = PotionomicsPotionStability.UNSTABLE
        elif cumulative_delta >= 0.05:
            self.current_stability = PotionomicsPotionStability.STABLE
        elif cumulative_delta > 0:
            self.current_stability = PotionomicsPotionStability.VERYSTABLE
        else:
            self.current_stability = PotionomicsPotionStability.PERFECT
        return cumulative_delta

    def render(self):
        raise NotImplementedError()

    def _get_item_names_from_idx(self) -> List[str]:
        return [self._action_to_ingredient[x].name for x in self.current_ingredients]

    def _get_current_ingredient_traits(self) -> np.ndarray:
        """Get the traits of all ingredients in the cauldron.

        :return: An array of the traits of all ingredients in the cauldron.
        Its shape is expected to be (N, 5) where N represents the number of ingredients currently in the cauldron.
        :rtype: np.ndarray
        """

        ingredients: List[int] = []
        for ingredient_idx in self.current_ingredients:
            ingredient: PotionomicsIngredient = self._action_to_ingredient[
                ingredient_idx
            ]
            ingredients.append(
                [
                    ingredient.total_magimin_value,
                    ingredient.trait_taste,
                    ingredient.trait_sensation,
                    ingredient.trait_smell,
                    ingredient.trait_sight,
                    ingredient.trait_hearing,
                ]
            )
        return np.array(ingredients, order="C")

    def calculate_potion_traits(self):
        """Calculate the traits that will be applied to the potion.

        Traits are likely calculated with the following steps:
            1. Non-neutral traits on ingredients with lower magimin totals are overridden by non-neutral traits with higher magimin totals.
            2. If two or more ingredients have the same magimin total, any overlapping traits are determined by uniform sampling. What remains unclear is whether multiple copies of a particular ingredient would influence the sampling or not.
        """

        current_ingredient_traits = self._get_current_ingredient_traits()
        unique_magimin_values = np.unique(
            current_ingredient_traits[:, 0]
        )  # Auto-sorts as least-to-greatest
        for magimin_value in unique_magimin_values:
            sub_matrix = current_ingredient_traits[
                current_ingredient_traits[:, 0] == magimin_value
            ][:, 1:]
        for idx, col in enumerate(sub_matrix[:, 1:].T):
            col = col[col != 0]
            if col.size != 0:
                self.potion_traits[idx] = np.random.choice(col, 1)
            else:
                self.potion_traits[idx] = 0

    def calculate_stability_bonus(self) -> float:
        """Calculate the stability bonus reward.

        Uses the following equation:
            `y = 1.661 * log_10(int(x) + 1e-8)`
        Which would have the following values for each stability value:
        |stability | x | y |
        |:-:|:-:|:-:|
        | CANNOTMAKE | 0 | -13.288    |
        | UNSTABLE   | 1 |   7.214e-9 |
        | STABLE     | 2 |   0.500    |
        | VERYSTABLE | 3 |   0.793    |
        | PERFECT    | 4 |   1.000    |

        :return: A floating-point value that represents the bonus reward for
        creating high-stability potions.
        :rtype: float
        """

        # The reason 1.661 is chosen is because we want Perfect ratio potions to give a reward of 1 (1.661*log_10(4) = 1)
        return 1.661 * np.log10(int(self.current_stability) + 1e-8)

    def _calculate_reward_function(self) -> torch.Tensor:
        """Internal function that calculates the reward.

        :return: The reward to give the agent for the current time-step.
        :rtype: float
        """

        cumulative_delta = torch.tensor(self.calculate_current_stability())
        reward = 0.0
        if self.cauldron.current_num_items < 2:
            return -1
        else:
            self.calculate_potion_rank_and_price()
            # The higher stability, the better
            stability_bonus = self.calculate_stability_bonus()
            reward = (
                (1.0 - cumulative_delta) * (self.cauldron.get_percent_full_magamin())
            ) + stability_bonus  # + price_vs_cost]
            reward = float(np.clip(reward, -1, 1))
        return reward

    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> None:
        """Reset the environment.

        :param seed: An integer that 'seeds' the random number generator, defaults to None
        :type seed: Optional[int], optional
        :param options: Additional options that can be passed into the function, defaults to None
        :type options: Dict[str, Any], optional
        """

        super().reset(seed=seed)
        self.current_ingredients = []
        self.current_stability = PotionomicsPotionStability.CANNOTMAKE
        self.potion_tier = None
        self.current_base_price = 0
        self.cost_of_items = 0
        self.potion_traits: np.ndarray = np.zeros((5,))
        self.num_ingredients: np.ndarray = np.ascontiguousarray(
            np.random.randint(
                low=1,
                high=self.cauldron.max_num_items_allowed + 1,
                size=len(self.all_ingredients) + 1,
            )
        )
        self.num_ingredients[0] = 0  # Set the None action to have 0
        # NOTE: We need to execute model_copy() because otherwise the information isn't cleared out
        # self.cauldron: PotionomicsCauldron = self.all_cauldrons[
        #     np.random.choice(len(self.all_recipes), 1)[0]
        # ].model_copy(deep=True)
        self.cauldron: PotionomicsCauldron = self.all_cauldrons[-1]
        self.cauldron.setup()
        # self.recipe = self.all_recipes[np.random.choice(len(self.all_recipes), 1)[0]]
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[List[Union[int, float]], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Perform a step in the environment.

        :param action: The action the agent selected. In this case, an action corresponds to an ingredient that the agent selects to put into the cauldron.
        :type action: int
        :return: A tuple that contains the state information after the action is performed, the reward, whether the episode has ended, whether the episode was truncated, and debug information.
        :rtype: Tuple[List[Union[int, float]], SupportsFloat, bool, bool, Dict[str, Any]]
        """

        terminated: bool = False
        legal_move: int = 1
        if self._action_to_ingredient[action]:
            legal_move = self.insert_item(index_of_item_to_insert=action)
        terminate_episode = (
            (action is None) or (not legal_move) or (self.cauldron.is_full())
        )
        if terminate_episode:
            terminated = True
        if legal_move:
            reward = self._calculate_reward_function()
        else:
            reward = -1
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def _get_obs(self) -> List[Union[int, float]]:
        observation = []
        ### Ingredient Information ###
        observation.extend(self.num_ingredients.tolist())
        ### Cauldron Information ###
        observation.extend(list(self.calculate_current_magimin_ratios()))
        observation.extend(self.recipe.magimin_ratios)
        observation.append(self.cauldron.current_total_magimin_amount)
        observation.append(self.cauldron.max_total_magimin_allowed)
        observation.append(self.cauldron.current_num_items)
        observation.append(self.cauldron.max_num_items_allowed)
        ### Potion Information ###
        observation.append(int(self.current_stability))
        observation.append(int(self.potion_tier) if self.potion_tier else -1)
        return observation

    def _get_info(self):
        return {
            "current_cauldron": self.cauldron,
            "current_recipe": self.recipe,
            "item_names": self._get_item_names_from_idx(),
            "current_recipe_ratios": self.calculate_current_magimin_ratios(),
            "current_stability": self.current_stability,
            "current_tier": self.potion_tier,
            "current_traits": self.potion_traits,
            "current_base_price": self.current_base_price,
            "current_cost": self.cost_of_items,
        }


def get_env() -> PotionomicsEnvironment:
    potionomics_items_file = "./potionomics_env/data/potionomics_ingredients.csv"
    potionomics_recipes_file = "./potionomics_env/data/potionomics_recipes.csv"
    potionomics_cauldrons_file = "./potionomics_env/data/potionomics_cauldrons.csv"
    all_ingredients: List[PotionomicsIngredient] = []
    with open(potionomics_items_file, "r") as _file:
        csv_data = [line for line in csv.DictReader(_file)]
        for entry in csv_data:
            item = PotionomicsIngredient(**entry)
            all_ingredients.append(item)

    all_recipes: List[PotionomicsPotionRecipe] = []
    with open(potionomics_recipes_file, "r") as _file:
        csv_data = [line for line in csv.DictReader(_file)]
        for entry in csv_data:
            item = PotionomicsPotionRecipe(**entry)
            item.calculate_magimin_ratios()
            item.calculate_magimin_ratios_int()
            all_recipes.append(item)

    all_cauldrons: List[PotionomicsCauldron] = []
    with open(potionomics_cauldrons_file, "r") as _file:
        csv_data = [line for line in csv.DictReader(_file)]
        for entry in csv_data:
            item = PotionomicsCauldron(**entry)
            item.setup()
            all_cauldrons.append(item)

    env = PotionomicsEnvironment(
        all_ingredients=all_ingredients,
        all_recipes=all_recipes,
        all_cauldrons=all_cauldrons,
    )
    return env
