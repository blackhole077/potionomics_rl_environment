from collections import deque
from collections import namedtuple
from itertools import count
import math
from pathlib import Path
import random

import logging
from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from potionomics_env.main import get_env
from potionomics_env.potionomics_test_cases import (
    generate_corsac_contest_potions,
    generate_finn_contest_potions,
    generate_roxanne_contest_potions,
    generate_anuberia_contest_potions,
    generate_robin_contest_potions,
)
from potionomics_env.schemas import PotionomicsPotion

logger = logging.getLogger(__name__)
logging.addLevelName(30, "SUCCESS")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV = get_env()
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward")
)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: Union[str, Path]) -> None:
        _state_dict = torch.load(path, map_location=device)
        self.load_state_dict(_state_dict, strict=True)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state: torch.Tensor, policy_net: DQN):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1

    # Use the environment's action masking function
    action_mask = ENV._calculate_action_mask()
    valid_actions = [i for i, valid in enumerate(action_mask) if valid]

    if sample > eps_threshold:
        with torch.no_grad():
            q_values = policy_net(state)
            # Mask invalid actions by setting their Q-values to -inf
            mask = torch.tensor(
                action_mask, device=device, dtype=torch.bool
            ).unsqueeze(0)
            # Clone the Q-values and apply the mask
            masked_q_values = q_values.clone()
            masked_q_values[~mask] = float("-inf")
            action = masked_q_values.max(1)[1].view(1, 1)
            return action
    else:
        action = random.choice(valid_actions)
        return torch.tensor([[action]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(
            non_final_next_states
        ).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def run_rl_loop(
    policy_net: DQN,
    target_net: DQN,
    memory: ReplayMemory,
    num_episodes: int = 100,
    is_training: bool = True,
) -> None:
    if not is_training:
        policy_net.eval()
        target_net.eval()
        _rewards = []
    HIGHEST_REWARD: float = -1.0
    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get it's state
        state, info = ENV.reset()
        state = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        for t in count():
            action = select_action(state, policy_net)
            # Perform the action in the environment and get the next state,
            # reward, and done flag
            observation, reward, terminated, truncated, info = ENV.step(
                action.item()
            )
            # Convert the observation to a tensor
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
            if is_training:
                if reward > 0 and reward >= HIGHEST_REWARD:
                    logger.log(30, f"Reward: {reward.item()}")
                    logger.log(30, f"Cauldron: {info['current_cauldron'].name}")
                    logger.log(
                        30, f"Recipe Name: {info['current_recipe'].name}"
                    )
                    logger.log(30, f"Ratios: {info['current_recipe_ratios']}")
                    logger.log(30, f"Stability: {info['current_stability']}")
                    logger.log(30, f"Traits: {info['current_traits']}")
                    logger.log(30, f"Tier: {info['current_tier']}")
                    logger.log(
                        30, f"Expected Profit: {info['expected_profit']}"
                    )
                    logger.log(30, f"Cost: {info['current_cost']}")
                    logger.log(30, f"Ingredients: {info['item_names']}")
                    HIGHEST_REWARD = reward
                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

            else:
                if done:
                    _rewards.append(reward.item())

            if done:
                break
    if not is_training:
        logger.log(35, f"Rewards: {_rewards}")


def select_action_greedy(state: torch.Tensor, policy_net: DQN):
    action_mask = ENV._calculate_action_mask()
    with torch.no_grad():
        q_values = policy_net(state)
        # Mask invalid actions by setting their Q-values to -inf
        mask = torch.tensor(
            action_mask, device=device, dtype=torch.bool
        ).unsqueeze(0)
        # Clone the Q-values and apply the mask
        masked_q_values = q_values.clone()
        masked_q_values[~mask] = float("-inf")
        action = masked_q_values.max(1)[1].view(1, 1)
        return action


def contest_loop(contest_potion: PotionomicsPotion, num_attempts: int = 100):
    logger.log(30, f"Contest Potion: {contest_potion}")
    for i_episode in tqdm(range(num_attempts)):
        # Initialize the environment and get it's state
        state, info = ENV.reset()
        state = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        for t in count():
            action = select_action_greedy(state, policy_net)
            # Perform the action in the environment and get the next state,
            # reward, and done flag
            observation, reward, terminated, truncated, info = ENV.step(
                action.item()
            )
            # Convert the observation to a tensor
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)
            if done:
                _potion: PotionomicsPotion = info["final_potion"]
                if _potion is not None and _potion.contest_potion_comparison(
                    contest_potion
                ):
                    logger.log(
                        35,
                        f"\nPotion {_potion} Passed the contest requirements\nTotal Attempts: {i_episode}",
                    )
                    return True
                break  # Exit loop when done
            else:
                # Move to the next state
                state = next_state


if __name__ == "__main__":
    print("Starting RL training...")
    # Get number of actions from gym action space
    n_actions = ENV.action_space.n
    # Get the number of state observations
    state, info = ENV.reset()
    n_observations = len(state)
    logger.debug(
        f"Number of Actions: {n_actions}\tNumber of Observations: {n_observations}"
    )
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)

    steps_done = 0

    if torch.cuda.is_available():
        num_episodes = 25000
    else:
        num_episodes = 50

    # run_rl_loop(policy_net, target_net, memory, num_episodes)
    # logger.log(30, "Saving model to path")
    path = "potionomics_agent.pth"
    policy_net.load(path)
    logger.log(35, "Testing RL Agent...")
    roxanne_contest_potions: List[PotionomicsPotion] = (
        generate_roxanne_contest_potions()
    )
    policy_net.eval()
    target_net.eval()
    for contest_potion in roxanne_contest_potions:
        index, _ = ENV.get_potion_recipe_from_name(contest_potion.potion_name)
        ENV.update_environment(
            ["task=roxanne_contest", f"recipe.recipe_index={index}"]
        )
        contest_loop(contest_potion, 1000)
    corsac_contest_potions: List[PotionomicsPotion] = (
        generate_corsac_contest_potions()
    )
    for contest_potion in corsac_contest_potions:
        index, _ = ENV.get_potion_recipe_from_name(contest_potion.potion_name)
        ENV.update_environment(
            ["task=corsac_contest", f"recipe.recipe_index={index}"]
        )
        contest_loop(contest_potion, 1000)
    finn_contest_potions: List[PotionomicsPotion] = (
        generate_finn_contest_potions()
    )
    for contest_potion in finn_contest_potions:
        index, _ = ENV.get_potion_recipe_from_name(contest_potion.potion_name)
        ENV.update_environment(
            ["task=finn_contest", f"recipe.recipe_index={index}"]
        )
        contest_loop(contest_potion, 1000)
    anuberia_contest_potions: List[PotionomicsPotion] = (
        generate_anuberia_contest_potions()
    )
    for contest_potion in anuberia_contest_potions:
        index, _ = ENV.get_potion_recipe_from_name(contest_potion.potion_name)
        ENV.update_environment(
            ["task=anuberia_contest", f"recipe.recipe_index={index}"]
        )
        contest_loop(contest_potion, 1000)
    robin_contest_potions: List[PotionomicsPotion] = (
        generate_robin_contest_potions()
    )
    for contest_potion in robin_contest_potions:
        index, _ = ENV.get_potion_recipe_from_name(contest_potion.potion_name)
        ENV.update_environment(
            ["task=robin_contest", f"recipe.recipe_index={index}"]
        )
        contest_loop(contest_potion, 1000)
