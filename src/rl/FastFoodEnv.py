import random

import gymnasium as gym
import numpy as np
import simpy
from gymnasium import spaces

from src.config import *
from src.sim.processes import FoodItem, customer_arrivals, inventory_manager
from src.sim.restaurant import FastFoodRestaurant


class FastFoodEnv(gym.Env):
    """
    A Reinforcement Learning environment for our Fast Food Sim.
    Includes Observation Scaling, Reward Scaling, Seed Support,
    and Enhanced AI "Vision" (Time and Active Cashiers).
    """

    def __init__(self):
        super(FastFoodEnv, self).__init__()

        # Staffing Layout (Optimized via Optuna)
        self.num_cashiers = 4
        self.num_burger_cooks = 2
        self.num_fries_cooks = 1

        # --- THE ACTION SPACE ---
        # 0: Do nothing, 1: Cook Burger, 2: Cook Fries, 3: Cook Both
        self.action_space = spaces.Discrete(4)

        # --- THE OBSERVATION SPACE ---
        # Normalized inputs are mathematically easier for Neural Networks to process.
        # [Queue, Burger Inv, Fries Inv, Idle Burger %, Idle Fries %, Time %, Busy Cashiers %]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

    def action_masks(self):
        """Calculates which actions are physically possible based on current staff."""
        idle_burgers = self.num_burger_cooks - self.restaurant.burger_cook.count
        idle_fries = self.num_fries_cooks - self.restaurant.fries_cook.count

        can_cook_burger = idle_burgers > 0
        can_cook_fries = idle_fries > 0
        can_cook_both = can_cook_burger and can_cook_fries

        return np.array(
            [True, can_cook_burger, can_cook_fries, can_cook_both], dtype=bool
        )

    def rl_cook_burger(self):
        """Task triggered by AI to occupy a burger cook and produce food."""
        with self.restaurant.burger_cook.request() as req:
            yield req
            yield self.env.timeout(
                random.triangular(BURGER_MIN, BURGER_MAX, BURGER_MODE)
            )
            for _ in range(BURGER_BATCH_SIZE):
                self.restaurant.burger_shelf.put(FoodItem(self.env.now))

    def rl_cook_fries(self):
        """Task triggered by AI to occupy a fries cook and produce fries."""
        with self.restaurant.fries_cook.request() as req:
            yield req
            yield self.env.timeout(FRIES_TIME)
            for _ in range(FRIES_BATCH_SIZE):
                self.restaurant.fries_shelf.put(FoodItem(self.env.now))

    def reset(self, seed=None, options=None):
        """Starts a brand new 1-hour shift with proper seed support."""
        # Standard Gymnasium seeding
        super().reset(seed=seed)

        # Seed both random libraries to ensure deterministic behavior if a seed is provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.env = simpy.Environment()
        self.restaurant = FastFoodRestaurant(
            self.env, self.num_cashiers, self.num_burger_cooks, self.num_fries_cooks
        )

        self.stats = {
            "wait_times": [],
            "wasted_burgers": [],
            "wasted_fries": [],
            "balked": [],
            "reneged": [],
            "captured_revenue": [],
            "lost_revenue": [],
        }

        # Launch the customer arrival process
        self.env.process(customer_arrivals(self.env, self.restaurant, self.stats))
        self.env.process(inventory_manager(self.env, self.restaurant, self.stats))
        # Reset financial delta trackers
        self.last_revenue = 0
        self.last_waste_cost = 0
        self.last_lost_revenue = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """
        Gathers and scales 7 observations to a [0, 1] range.
        """
        # Divide by 'Max Expected' values to keep inputs small for the Neural Network
        queue_len = len(self.restaurant.cashier.queue) / 20.0
        burger_inv = len(self.restaurant.burger_shelf.items) / 30.0
        fries_inv = len(self.restaurant.fries_shelf.items) / 30.0

        # Scale idle staff as a percentage of total capacity
        idle_burgers = (
            self.num_burger_cooks - self.restaurant.burger_cook.count
        ) / self.num_burger_cooks
        idle_fries = (
            self.num_fries_cooks - self.restaurant.fries_cook.count
        ) / self.num_fries_cooks

        # --- THE NEW OBSERVATIONS ---
        # 1. Time Percentage (0.0 at start, 1.0 at end of shift)
        time_pct = self.env.now / SIM_TIME

        # 2. Busy Cashiers Percentage (0.0 if empty, 1.0 if all cashiers are taking orders)
        busy_cashiers = self.restaurant.cashier.count / self.num_cashiers

        obs = np.array(
            [
                queue_len,
                burger_inv,
                fries_inv,
                idle_burgers,
                idle_fries,
                time_pct,
                busy_cashiers,
            ],
            dtype=np.float32,
        )
        # Ensure we never exceed the Box(0, 1) limits
        return np.clip(obs, 0.0, 1.0)

    def step(self, action):
        """The AI makes a move, and we run the clock forward 10 seconds."""
        # 1. Execute valid actions (checked via Action Masking in train_ai.py)
        if action == 1 or action == 3:
            self.env.process(self.rl_cook_burger())
        if action == 2 or action == 3:
            self.env.process(self.rl_cook_fries())

        # 2. Run simulation 10 seconds forward
        self.env.run(until=self.env.now + 10)

        # 3. Reward Calculation using Deltas
        current_revenue = sum(self.stats["captured_revenue"])
        current_waste = (len(self.stats["wasted_burgers"]) * COST_WASTED_BURGER) + (
            len(self.stats["wasted_fries"]) * COST_WASTED_FRIES
        )
        current_lost = sum(self.stats["lost_revenue"])

        # Delta calculation (What happened in the LAST 10 seconds)
        delta_rev = current_revenue - self.last_revenue
        delta_waste = current_waste - self.last_waste_cost
        delta_lost = current_lost - self.last_lost_revenue

        # --- REWARD SCALING ---
        # Scaling factors stabilize training by keeping rewards in a manageable range.
        scale_factor = 10.0
        reward = (delta_rev - delta_waste - delta_lost) / scale_factor

        # Update trackers for next step
        self.last_revenue = current_revenue
        self.last_waste_cost = current_waste
        self.last_lost_revenue = current_lost

        # 4. Termination Check (Shift ends at 1 hour)
        terminated = self.env.now >= SIM_TIME
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, {}
