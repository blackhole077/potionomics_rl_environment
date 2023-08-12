from enum import IntEnum


class PotionomicsIngredientType(IntEnum):
    """An enum that describes the different types of ingredients used in Potionomics."""

    UNKNOWN = 0
    BONE = 1
    BUG = 2
    ESSENCE = 3
    FISH = 4
    FLESH = 5
    FLOWER = 6
    FRUIT = 7
    FUNGUS = 8
    GEM = 9
    MINERAL = 10
    ORE = 11
    PLANT = 12
    MANA = 13
    SLIME = 14


class PotionomicsPotionRecipeType(IntEnum):
    """An enum that describes the different categories of potions that can be made in Potionomics."""

    UNKNOWN = 0
    BASIC = 1
    TONIC = 2
    ENHANCER = 3
    CURE = 4


class PotionomicsPotionTier(IntEnum):
    """An enum that describes the different tiers of potions that can be made in Potionomics."""

    MINOR = 0
    COMMON = 1
    GREATER = 2
    GRAND = 3
    SUPERIOR = 4
    MASTERWORK = 5


class PotionomicsPotionStability(IntEnum):
    """An enum that describes the different level of stabilities a potion can have in Potionomics."""

    CANNOTMAKE = 0
    UNSTABLE = 1
    STABLE = 2
    VERYSTABLE = 3
    PERFECT = 4
