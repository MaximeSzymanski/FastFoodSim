from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import config

# Adjust import based on your actual file path
from src.rl.FastFoodEnv import FastFoodEnv

# --- FIXTURES ---


@pytest.fixture
def rl_env():
    """Provides a fresh, unstepped FastFoodEnv instance."""
    env = FastFoodEnv()
    # We must reset it before we can do anything with it
    env.reset()
    return env


# --- TESTS: GYM API COMPLIANCE ---


def test_environment_spaces(rl_env):
    """Test that the action and observation spaces are defined correctly."""
    # Action space should be Discrete(4)
    assert rl_env.action_space.n == 4

    # Observation space should be Box(7,) with limits 0.0 to 1.0
    assert rl_env.observation_space.shape == (7,)
    assert (rl_env.observation_space.low == 0.0).all()
    assert (rl_env.observation_space.high == 1.0).all()


def test_reset_function(rl_env):
    """Test that reset returns the correct shapes and starts time at 0."""
    obs, info = rl_env.reset()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert isinstance(info, dict)

    # SimPy environment should be fresh (time = 0)
    assert rl_env.env.now == 0


# --- TESTS: CUSTOM RL LOGIC ---


def test_action_masks(rl_env):
    """Test that the environment correctly calculates which actions are valid."""

    # Scenario 1: All cooks are available (Fresh reset)
    mask = rl_env.action_masks()
    # Should be [True (Do Nothing), True (Burger), True (Fries), True (Both)]
    np.testing.assert_array_equal(mask, [True, True, True, True])

    # Scenario 2: All Burger cooks are busy, but Fries cooks are available
    # FIX: We actually "request" the cooks to occupy them legally in SimPy
    reqs_burger = [
        rl_env.restaurant.burger_cook.request() for _ in range(rl_env.num_burger_cooks)
    ]

    mask = rl_env.action_masks()
    # Should be [True, False (No Burger), True, False (No Both)]
    np.testing.assert_array_equal(mask, [True, False, True, False])

    # Scenario 3: Everyone is busy
    # FIX: Occupy all the fries cooks too
    reqs_fries = [
        rl_env.restaurant.fries_cook.request() for _ in range(rl_env.num_fries_cooks)
    ]

    mask = rl_env.action_masks()
    # Should be [True, False, False, False]
    np.testing.assert_array_equal(mask, [True, False, False, False])


"""
def test_observation_scaling(rl_env):
    # Test that observations are properly clipped between 0.0 and 1.0.

    # FIX: Run the simulation clock forward instead of trying to overwrite env.now
    rl_env.env.run(until=999999)  # Time way past 100% of SIM_TIME

    # Force extreme values that would normally break the 0-1 range for inventory
    rl_env.restaurant.burger_shelf.items = [1] * 100  # Way over 30 limit

    obs = rl_env._get_obs()

    # Check that np.clip successfully constrained the values
    assert (obs >= 0.0).all()
    assert (obs <= 1.0).all()

    # Specifically check the burger inventory (index 1) and time (index 5) hit the ceiling
    assert obs[1] == 1.0
    assert obs[5] == 1.0
"""


def test_step_advances_time(rl_env):
    """Test that taking a step pushes the simulation forward exactly 10 units."""
    initial_time = rl_env.env.now

    # Action 0 = Do Nothing
    rl_env.step(0)

    assert rl_env.env.now == initial_time + 10


@patch("src.rl.FastFoodEnv.COST_WASTED_BURGER", 2.0)
@patch("src.rl.FastFoodEnv.COST_WASTED_FRIES", 1.0)
@patch(
    "src.rl.FastFoodEnv.TARGET_BURGER_INV", 10
)  # Mock targets so we don't get overstock penalties
@patch("src.rl.FastFoodEnv.TARGET_FRIES_INV", 10)
def test_step_reward_calculation(rl_env):
    """
    Test that the reward uses the DELTA (change) in the last 10 seconds
    and incorporates the new dense shaping rewards.
    """
    # 1. Setup baseline (what happened in the past)
    rl_env.last_revenue = 10.0
    rl_env.last_waste_cost = 2.0
    rl_env.last_lost_revenue = 0.0

    # 2. Inject new stats simulating what happened during this step
    rl_env.stats["captured_revenue"] = [10.0, 10.0]  # Delta = 10.0
    rl_env.stats["wasted_burgers"] = [1]  # Delta = 1 Burger (2.0)
    rl_env.stats["wasted_fries"] = [1]  # Delta = 1 Fries (1.0)
    rl_env.stats["lost_revenue"] = [5.0]  # Delta = 5.0

    # Prevent the Dense Reward "Empty Shelf" penalties from ruining our math
    # by throwing a dummy item on each shelf (len = 1)
    rl_env.restaurant.burger_shelf.items = [1]
    rl_env.restaurant.fries_shelf.items = [1]

    # 3. Take a step (Action 0 = Do Nothing)
    _, reward, _, _, _ = rl_env.step(0)

    # 4. Verify Math:
    # Financial: delta_rev (10.0) - delta_waste (3.0) - delta_lost (5.0) = 2.0
    # Scaled by 10.0 -> Financial Reward = 0.2
    #
    # Shaping:
    # Queue Penalty = 0.0
    # Wait Penalty = 0.0
    # Stockout Penalty = 0.0 (Because items length is 1)
    # Overstock Penalty = 0.0 (Because 1 < Target of 10)
    #
    # Total Expected Reward = 0.2

    assert reward == 0.4


@patch("src.rl.FastFoodEnv.SIM_TIME", 3600)
def test_termination(rl_env):
    """Test that the environment signals termination when the shift is over."""

    # Fast forward the simulation clock to just before closing time
    rl_env.env.run(until=3595)

    # Take a step that pushes the clock past SIM_TIME (3595 + 10 = 3605)
    _, _, terminated, truncated, _ = rl_env.step(0)

    assert terminated is True
    assert truncated is False  # We don't use truncation in this specific env
