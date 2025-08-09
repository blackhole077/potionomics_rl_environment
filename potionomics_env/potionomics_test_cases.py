"""This module contains test cases to evaluate the performance of the agent with
potions from the game's storyline. For simplicity, the test cases replicate the
conditions needed for the achievement "No Contest", where the player skips the
haggling minigame by producing a potion that is valued higher than the opponents
potion.
"""

from typing import List
from potionomics_env.potionomics_enum import (
    PotionomicsPotionRecipeType,
    PotionomicsPotionTier,
)
from potionomics_env.schemas import PotionomicsPotion


def generate_roxanne_contest_potions() -> List[PotionomicsPotion]:
    roxanne_contest_potion_one = PotionomicsPotion(
        Name="Health_Potion",
        Type=PotionomicsPotionRecipeType.BASIC,
        Rank=PotionomicsPotionTier.COMMON,
        Stars=0,
        Price=72,
    )
    roxanne_contest_potion_two = PotionomicsPotion(
        Name="Fire_Tonic",
        Type=PotionomicsPotionRecipeType.TONIC,
        Rank=PotionomicsPotionTier.COMMON,
        Stars=0,
        Price=72,
    )
    roxanne_contest_potion_three = PotionomicsPotion(
        Name="Mana_Potion",
        Type=PotionomicsPotionRecipeType.BASIC,
        Rank=PotionomicsPotionTier.COMMON,
        Stars=0,
        Price=72,
    )
    return [
        roxanne_contest_potion_one,
        roxanne_contest_potion_two,
        roxanne_contest_potion_three,
    ]


def generate_corsac_contest_potions() -> List[PotionomicsPotion]:
    corsac_contest_potion_one = PotionomicsPotion(
        Name="Ice_Tonic",
        Type=PotionomicsPotionRecipeType.TONIC,
        Rank=PotionomicsPotionTier.GREATER,
        Stars=0,
        Price=215,
    )
    corsac_contest_potion_two = PotionomicsPotion(
        Name="Sight_Enhancer",
        Type=PotionomicsPotionRecipeType.ENHANCER,
        Rank=PotionomicsPotionTier.GREATER,
        Stars=0,
        Price=215,
    )
    corsac_contest_potion_three = PotionomicsPotion(
        Name="Speed_Potion",
        Type=PotionomicsPotionRecipeType.BASIC,
        Rank=PotionomicsPotionTier.GREATER,
        Stars=0,
        Price=215,
    )
    return [
        corsac_contest_potion_one,
        corsac_contest_potion_two,
        corsac_contest_potion_three,
    ]


def generate_finn_contest_potions() -> List[PotionomicsPotion]:
    finn_contest_potion_one = PotionomicsPotion(
        Name="Poison_Cure",
        Type=PotionomicsPotionRecipeType.CURE,
        Rank=PotionomicsPotionTier.GRAND,
        Stars=0,
        Price=450,
    )
    finn_contest_potion_two = PotionomicsPotion(
        Name="Thunder_Tonic",
        Type=PotionomicsPotionRecipeType.TONIC,
        Rank=PotionomicsPotionTier.GRAND,
        Stars=0,
        Price=510,
    )
    finn_contest_potion_three = PotionomicsPotion(
        Name="Stamina_Potion",
        Type=PotionomicsPotionRecipeType.BASIC,
        Rank=PotionomicsPotionTier.GRAND,
        Stars=0,
        Price=495,
    )
    return [
        finn_contest_potion_one,
        finn_contest_potion_two,
        finn_contest_potion_three,
    ]


def generate_anuberia_contest_potions() -> List[PotionomicsPotion]:
    anuberia_contest_potion_one = PotionomicsPotion(
        Name="Silence_Cure",
        Type=PotionomicsPotionRecipeType.CURE,
        Rank=PotionomicsPotionTier.SUPERIOR,
        Stars=0,
        Price=1200,
    )
    anuberia_contest_potion_two = PotionomicsPotion(
        Name="Tolerance_Potion",
        Type=PotionomicsPotionRecipeType.BASIC,
        Rank=PotionomicsPotionTier.SUPERIOR,
        Stars=0,
        Price=1440,
    )
    anuberia_contest_potion_three = PotionomicsPotion(
        Name="Insight_Enhancer",
        Type=PotionomicsPotionRecipeType.ENHANCER,
        Rank=PotionomicsPotionTier.SUPERIOR,
        Stars=0,
        Price=1530,
    )
    return [
        anuberia_contest_potion_one,
        anuberia_contest_potion_two,
        anuberia_contest_potion_three,
    ]


def generate_robin_contest_potions() -> List[PotionomicsPotion]:
    robin_contest_potion_one = PotionomicsPotion(
        Name="Radiation_Tonic",
        Type=PotionomicsPotionRecipeType.TONIC,
        Rank=PotionomicsPotionTier.MASTERWORK,
        Stars=0,
        Price=3800,
    )
    robin_contest_potion_two = PotionomicsPotion(
        Name="Curse_Cure",
        Type=PotionomicsPotionRecipeType.CURE,
        Rank=PotionomicsPotionTier.MASTERWORK,
        Stars=0,
        Price=3600,
    )
    robin_contest_potion_three = PotionomicsPotion(
        Name="Seeking_Enhancer",
        Type=PotionomicsPotionRecipeType.ENHANCER,
        Rank=PotionomicsPotionTier.MASTERWORK,
        Stars=0,
        Price=3750,
    )
    return [
        robin_contest_potion_one,
        robin_contest_potion_two,
        robin_contest_potion_three,
    ]
