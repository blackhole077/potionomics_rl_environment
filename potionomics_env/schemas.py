from typing import Optional, List
from pydantic import BaseModel, Field

from potionomics_env.potionomics_enum import (
    PotionomicsIngredientType,
    PotionomicsPotionRecipeType,
)
import numpy as np


class PotionomicsIngredient(BaseModel):
    """A Pydantic schema that describes an ingredient in Potionomics.

    Attributes:
        name [str]: The name of the ingredient.
        ingredient_type [PotionomicsIngredientType]: An integer corresponding to what category the ingredient belongs to.
        a_magimin_value [int]: The amount of magimins in the 'A' category.
        b_magimin_value [int]: The amount of magimins in the 'B' category.
        c_magimin_value [int]: The amount of magimins in the 'C' category.
        d_magimin_value [int]: The amount of magimins in the 'D' category.
        e_magimin_value [int]: The amount of magimins in the 'E' category.
        total_magimin_value [int]: The total amount of magimins the ingredient contains.
        item_price [int]: The price of the ingredient when purchased from Quinn's store.
        trait_taste [int]: An integer that indicates whether it has negative, positive, or neutral taste (i.e., [-1, 1, 0])
        trait_sensation [int]: An integer that indicates whether it has negative, positive, or neutral sensation (i.e., [-1, 1, 0])
        trait_smell [int]: An integer that indicates whether it has negative, positive, or neutral smell (i.e., [-1, 1, 0])
        trait_sight [int]: An integer that indicates whether it has negative, positive, or neutral sight (i.e., [-1, 1, 0])
        trait_hearing [int]: An integer that indicates whether it has negative, positive, or neutral hearing (i.e., [-1, 1, 0])
    """

    name: str = Field(alias="Ingredient")
    ingredient_type: PotionomicsIngredientType = Field(alias="Type")
    a_magimin_value: int = Field(alias="A")
    b_magimin_value: int = Field(alias="B")
    c_magimin_value: int = Field(alias="C")
    d_magimin_value: int = Field(alias="D")
    e_magimin_value: int = Field(alias="E")
    total_magimin_value: int = Field(alias="Total")
    item_price: int = Field(alias="Price")
    trait_taste: int = Field(alias="Taste")
    trait_sensation: int = Field(alias="Sensation")
    trait_smell: int = Field(alias="Smell")
    trait_sight: int = Field(alias="Sight")
    trait_hearing: int = Field(alias="Hearing")

    def __repr__(self):
        return f"\
Name of Item: {self.name}\t[{self.ingredient_type}]\tPrice: {self.item_price}\n\
A: {self.a_magimin_value}\n\
B: {self.b_magimin_value}\n\
C: {self.c_magimin_value}\n\
D: {self.d_magimin_value}\n\
E: {self.e_magimin_value}\n\
Total: {self.total_magimin_value}\n\
"


class PotionomicsPotionRecipe(BaseModel):
    """A Pydantic schema that describes a potion recipe in Potionomics.

    Attributes:

        name [str]: The name of the recipe.

        potion_type [PotionomicsPotionRecipeType]: An integer corresponding to what category the potion belongs to.

        a_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'A' category.

        b_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'B' category.

        c_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'C' category.

        d_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'D' category.

        e_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'E' category.

        ratio_total Optional[int]: The sum of all magimin ratios. Used to calculate the ratios in floating-point value form.

        magimin_ratios Optional[np.ndarray]: An array that contains the magimin ratios in floating-point value form.
    """

    name: str = Field(alias="Name")
    potion_type: PotionomicsPotionRecipeType = Field(alias="Type")
    a_magimin_ratio_int: int = Field(alias="A")
    b_magimin_ratio_int: int = Field(alias="B")
    c_magimin_ratio_int: int = Field(alias="C")
    d_magimin_ratio_int: int = Field(alias="D")
    e_magimin_ratio_int: int = Field(alias="E")
    ratio_total: Optional[int] = Field(default=0)
    magimin_ratios_int: Optional[List[int]] = Field(default=[])
    magimin_ratios: Optional[List[float]] = Field(default=[])

    def calculate_magimin_ratios_int(self):
        """Compile the magimin ratios as integers.
        Used by get_percent_full_magamin
        """
        self.magimin_ratios_int = np.ascontiguousarray(
            np.array(
                [
                    self.a_magimin_ratio_int,
                    self.b_magimin_ratio_int,
                    self.c_magimin_ratio_int,
                    self.d_magimin_ratio_int,
                    self.e_magimin_ratio_int,
                ],
                dtype=int,
            ),
        )

    def calculate_magimin_ratios(self) -> None:
        """Convert the recipe ratios from integer values to floating-point values."""

        self.ratio_total = (
            self.a_magimin_ratio_int
            + self.b_magimin_ratio_int
            + self.c_magimin_ratio_int
            + self.d_magimin_ratio_int
            + self.e_magimin_ratio_int
        )
        self.magimin_ratios = np.array(
            [
                self.a_magimin_ratio_int / self.ratio_total,
                self.b_magimin_ratio_int / self.ratio_total,
                self.c_magimin_ratio_int / self.ratio_total,
                self.d_magimin_ratio_int / self.ratio_total,
                self.e_magimin_ratio_int / self.ratio_total,
            ]
        )

    def __repr__(self):
        return f"""
Name: {self.name}\t[{self.potion_type}]
Recipe: <{self.a_magimin_ratio_int}\t{self.b_magimin_ratio_int}\t{self.c_magimin_ratio_int}\t{self.d_magimin_ratio_int}\t{self.e_magimin_ratio_int}>
    """


class PotionomicsCauldron(BaseModel):
    """A Pydantic schema that describes an ingredient in Potionomics.

    Attributes:

        name [str]: The name of the ingredient.

        ingredient_type [PotionomicsIngredientType]: An integer corresponding to what category the ingredient belongs to.

        a_magimin_value [int]: The amount of magimins in the 'A' category.

        b_magimin_value [int]: The amount of magimins in the 'B' category.

        c_magimin_value [int]: The amount of magimins in the 'C' category.

        d_magimin_value [int]: The amount of magimins in the 'D' category.

        e_magimin_value [int]: The amount of magimins in the 'E' category.

        total_magimin_value [int]: The total amount of magimins the ingredient contains.

        item_price [int]: The price of the ingredient when purchased from Quinn's store.

        trait_taste [int]: An integer that indicates whether it has negative, positive, or neutral taste (i.e., [-1, 1, 0])

        trait_sensation [int]: An integer that indicates whether it has negative, positive, or neutral sensation (i.e., [-1, 1, 0])

        trait_smell [int]: An integer that indicates whether it has negative, positive, or neutral smell (i.e., [-1, 1, 0])

        trait_sight [int]: An integer that indicates whether it has negative, positive, or neutral sight (i.e., [-1, 1, 0])

        trait_hearing [int]: An integer that indicates whether it has negative, positive, or neutral hearing (i.e., [-1, 1, 0])
    """

    name: str = Field(alias="Name")
    max_num_items_allowed: int = Field(alias="Max Ingredients")
    max_total_magimin_allowed: int = Field(alias="Max Magimins")
    current_num_items: Optional[int] = Field(default=0)
    current_total_magimin_amount: Optional[int] = Field(default=0)
    current_a_magimin_amount: Optional[int] = Field(default=0)
    current_b_magimin_amount: Optional[int] = Field(default=0)
    current_c_magimin_amount: Optional[int] = Field(default=0)
    current_d_magimin_amount: Optional[int] = Field(default=0)
    current_e_magimin_amount: Optional[int] = Field(default=0)

    def setup(self):
        """Simple setup function that fills in values we don't expect in the CSV file with default values."""

        self.current_num_items = 0
        self.current_total_magimin_amount = 0
        self.current_a_magimin_amount = 0
        self.current_b_magimin_amount = 0
        self.current_c_magimin_amount = 0
        self.current_d_magimin_amount = 0
        self.current_e_magimin_amount = 0

    def is_full(self) -> bool:
        """Determine if the cauldron is full.

        :return: A flag that is set to true if the number of items equals or
        exceeds the cauldron contents.
        :rtype: bool
        """

        return self.current_num_items >= self.max_num_items_allowed

    def get_percent_full_magamin(self) -> float:
        """Get the amount of magamins in the cauldron as a percentage.

        :return: A floating-point value in the range [0, 1].
        :rtype: float
        """

        return float(self.current_total_magimin_amount) / float(
            self.max_total_magimin_allowed
        )

    def __repr__(self):
        return f"""
Number of items in cauldron: {self.current_num_items}/{self.max_num_items_allowed}
Total Magimin Amount:
A: {self.current_a_magimin_amount}
B: {self.current_b_magimin_amount}
C: {self.current_c_magimin_amount}
D: {self.current_d_magimin_amount}
E: {self.current_e_magimin_amount}
Total: {self.current_total_magimin_amount}/{self.max_total_magimin_allowed}
        """

    """A Pydantic schema that describes a potion recipe in Potionomics.

    Attributes:

        name [str]: The name of the recipe.

        potion_type [PotionomicsPotionRecipeType]: An integer corresponding to what category the potion belongs to.

        a_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'A' category.

        b_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'B' category.

        c_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'C' category.

        d_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'D' category.

        e_magimin_ratio_int [int]: An integer that indicates how much of the total magimin count should belong to the 'E' category.

        ratio_total Optional[int]: The sum of all magimin ratios. Used to calculate the ratios in floating-point value form.

        magimin_ratios Optional[np.ndarray]: An array that contains the magimin ratios in floating-point value form.
    """

    name: str = Field(alias="Name")
    potion_type: PotionomicsPotionRecipeType = Field(alias="Type")
    a_magimin_ratio_int: int = Field(alias="A")
    b_magimin_ratio_int: int = Field(alias="B")
    c_magimin_ratio_int: int = Field(alias="C")
    d_magimin_ratio_int: int = Field(alias="D")
    e_magimin_ratio_int: int = Field(alias="E")
    ratio_total: Optional[int] = Field(default=0)
    magimin_ratios_int: Optional[List[int]] = Field(default=[])
    magimin_ratios: Optional[List[float]] = Field(default=[])

    def calculate_magimin_ratios_int(self):
        """Compile the magimin ratios as integers.
        Used by get_percent_full_magamin
        """
        self.magimin_ratios_int = np.ascontiguousarray(
            np.array(
                [
                    self.a_magimin_ratio_int,
                    self.b_magimin_ratio_int,
                    self.c_magimin_ratio_int,
                    self.d_magimin_ratio_int,
                    self.e_magimin_ratio_int,
                ],
                dtype=int,
            ),
        )

    def calculate_magimin_ratios(self) -> None:
        """Convert the recipe ratios from integer values to floating-point values."""

        self.ratio_total = (
            self.a_magimin_ratio_int
            + self.b_magimin_ratio_int
            + self.c_magimin_ratio_int
            + self.d_magimin_ratio_int
            + self.e_magimin_ratio_int
        )
        self.magimin_ratios = np.array(
            [
                self.a_magimin_ratio_int / self.ratio_total,
                self.b_magimin_ratio_int / self.ratio_total,
                self.c_magimin_ratio_int / self.ratio_total,
                self.d_magimin_ratio_int / self.ratio_total,
                self.e_magimin_ratio_int / self.ratio_total,
            ]
        )

    def __repr__(self):
        return f"""
Name: {self.name}\t[{self.potion_type}]
Recipe: <{self.a_magimin_ratio_int}\t{self.b_magimin_ratio_int}\t{self.c_magimin_ratio_int}\t{self.d_magimin_ratio_int}\t{self.e_magimin_ratio_int}>
    """
