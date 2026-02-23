import os
import sys
import warnings

# Hide the Pygame welcome message
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import random

import gymnasium as gym
import numpy as np
import pygame
import simpy
from gymnasium import spaces

from src.config import *
from src.sim.processes import FoodItem, customer_arrivals, inventory_manager
from src.sim.restaurant import FastFoodRestaurant

# --- MODERNIZED VISUALIZER COLORS ---
BG_COLOR = (24, 24, 28)
FLOOR_COLOR = (40, 42, 48)
TEXT_COLOR = (245, 245, 245)
CUSTOMER_COLOR = (52, 152, 219)
CASHIER_COLOR = (155, 89, 182)
B_COOK_COLOR = (231, 76, 60)  # Red
F_COOK_COLOR = (241, 196, 15)  # Yellow
I_COOK_COLOR = (26, 188, 156)  # Mint/Turquoise for Ice Cream
IDLE_GRAY = (70, 75, 80)
WAITING_FOOD_COLOR = (46, 204, 113)
COUNTER_COLOR = (139, 90, 43)
TABLE_COLOR = (189, 195, 199)
UI_BAR_COLOR = (15, 15, 18)


class FastFoodEnv(gym.Env):
    """
    A Reinforcement Learning environment for our Fast Food Sim.
    Now uses pure financial logic, reputation penalties, and service bonuses!
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(FastFoodEnv, self).__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.title_font = None

        # Staffing Layout
        self.num_cashiers = 5
        self.num_burger_cooks = 4
        self.num_fries_cooks = 3
        self.num_ice_cream_cooks = 2

        # --- MODULAR ACTION SPACE ---
        self.action_space = spaces.MultiDiscrete([2, 2, 2])

        # --- SCALABLE OBSERVATION SPACE ---
        # 12 Inputs: [Queue, Time, Busy Cashiers, Inv B/F/I, Idle B/F/I, PENDING B/F/I]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

    def action_masks(self):
        idle_b = self.num_burger_cooks - self.restaurant.burger_cook.count
        idle_f = self.num_fries_cooks - self.restaurant.fries_cook.count
        idle_i = self.num_ice_cream_cooks - self.restaurant.ice_cream_cook.count

        can_cook_burger = idle_b > 0
        can_cook_fries = idle_f > 0
        can_cook_ice_cream = idle_i > 0

        return np.array(
            [True, can_cook_burger, True, can_cook_fries, True, can_cook_ice_cream],
            dtype=bool,
        )

    def rl_cook_burger(self):
        with self.restaurant.burger_cook.request() as req:
            yield req
            yield self.env.timeout(
                random.triangular(BURGER_MIN, BURGER_MAX, BURGER_MODE)
            )
            for _ in range(BURGER_BATCH_SIZE):
                self.restaurant.burger_shelf.put(FoodItem(self.env.now))

    def rl_cook_fries(self):
        with self.restaurant.fries_cook.request() as req:
            yield req
            yield self.env.timeout(FRIES_TIME)
            for _ in range(FRIES_BATCH_SIZE):
                self.restaurant.fries_shelf.put(FoodItem(self.env.now))

    def rl_cook_ice_cream(self):
        with self.restaurant.ice_cream_cook.request() as req:
            yield req
            yield self.env.timeout(ICE_CREAM_TIME)
            for _ in range(ICE_CREAM_BATCH_SIZE):
                self.restaurant.ice_cream_shelf.put(FoodItem(self.env.now))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.env = simpy.Environment()
        self.restaurant = FastFoodRestaurant(
            self.env,
            self.num_cashiers,
            self.num_burger_cooks,
            self.num_fries_cooks,
            num_ice_cream_cooks=self.num_ice_cream_cooks,
        )

        self.stats = {
            "wait_times": [],
            "wasted_burgers": [],
            "wasted_fries": [],
            "wasted_ice_cream": [],
            "balked": [],
            "reneged": [],
            "captured_revenue": [],
            "lost_revenue": [],
        }

        self.env.process(customer_arrivals(self.env, self.restaurant, self.stats))
        self.env.process(inventory_manager(self.env, self.restaurant, self.stats))

        # Reward tracking states
        self.last_revenue = 0
        self.last_waste_cost = 0
        self.last_lost_revenue = 0
        self.last_walkouts = 0  # NEW: For the Stick
        self.last_served = 0  # NEW: For the Carrot

        return self._get_obs(), {}

    def _get_obs(self):
        queue_len = len(self.restaurant.cashier.queue) / 20.0
        time_pct = self.env.now / SIM_TIME
        busy_cashiers = self.restaurant.cashier.count / self.num_cashiers

        # Inventories
        burger_inv = len(self.restaurant.burger_shelf.items) / 30.0
        fries_inv = len(self.restaurant.fries_shelf.items) / 30.0
        ice_cream_inv = len(self.restaurant.ice_cream_shelf.items) / 30.0

        # Idle Staff
        idle_b = (
            self.num_burger_cooks - self.restaurant.burger_cook.count
        ) / self.num_burger_cooks
        idle_f = (
            self.num_fries_cooks - self.restaurant.fries_cook.count
        ) / self.num_fries_cooks
        idle_i = (
            self.num_ice_cream_cooks - self.restaurant.ice_cream_cook.count
        ) / self.num_ice_cream_cooks

        # Pending Orders (Scaled by 30)
        pending_b = self.restaurant.pending_burgers / 30.0
        pending_f = self.restaurant.pending_fries / 30.0
        pending_i = self.restaurant.pending_ice_cream / 30.0

        obs = np.array(
            [
                queue_len,
                time_pct,
                busy_cashiers,
                burger_inv,
                fries_inv,
                ice_cream_inv,
                idle_b,
                idle_f,
                idle_i,
                pending_b,
                pending_f,
                pending_i,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)

    def step(self, action):
        if action[0] == 1:
            self.env.process(self.rl_cook_burger())
        if action[1] == 1:
            self.env.process(self.rl_cook_fries())
        if action[2] == 1:
            self.env.process(self.rl_cook_ice_cream())

        self.env.run(until=self.env.now + 10)

        # 1. Fetch current totals
        current_revenue = sum(self.stats["captured_revenue"])
        current_waste = (
            (len(self.stats["wasted_burgers"]) * COST_WASTED_BURGER)
            + (len(self.stats["wasted_fries"]) * COST_WASTED_FRIES)
            + (len(self.stats["wasted_ice_cream"]) * COST_WASTED_ICE_CREAM)
        )
        current_lost = sum(self.stats["lost_revenue"])
        current_walkouts = len(self.stats["balked"]) + len(self.stats["reneged"])
        current_served = len(self.stats["captured_revenue"])

        # 2. Calculate deltas (what happened in the last 10 seconds)
        delta_rev = current_revenue - self.last_revenue
        delta_waste = current_waste - self.last_waste_cost
        delta_lost = current_lost - self.last_lost_revenue
        delta_walkouts = current_walkouts - self.last_walkouts
        delta_served = current_served - self.last_served

        # 3. Base Financial Reward
        financial_reward = (delta_rev - delta_waste - delta_lost) / 10.0

        # 4. FIX 1: The Stick (Reputation Penalty)
        # Punish doing nothing. Walk-outs now carry a massive flat penalty beyond just lost revenue.
        reputation_penalty = delta_walkouts * 5.0

        # 5. FIX 2: The Carrot (Service Bonus)
        # Reward taking action. Give a small flat bonus for every successfully completed order.
        served_bonus = delta_served * 2.0

        # Grand Total Calculation (Dense Shaping Removed)
        total_reward = financial_reward - reputation_penalty + served_bonus

        # Update trackers
        self.last_revenue = current_revenue
        self.last_waste_cost = current_waste
        self.last_lost_revenue = current_lost
        self.last_walkouts = current_walkouts
        self.last_served = current_served

        terminated = self.env.now >= SIM_TIME
        truncated = False

        if self.render_mode in ["human", "rgb_array"]:
            self.render()

        return self._get_obs(), float(total_reward), terminated, truncated, {}

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 700))
            pygame.display.set_caption("AI Kitchen Manager Pro - Lunch Rush Edition")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16, bold=True)
            self.title_font = pygame.font.SysFont("Arial", 14, bold=True)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(BG_COLOR)

        # --- 1. DYNAMIC ARCHITECTURE ---
        b_table_w = max(150, self.num_burger_cooks * 70)
        f_table_w = max(150, self.num_fries_cooks * 70)
        i_table_w = max(150, self.num_ice_cream_cooks * 70)

        b_table_x = 80
        f_table_x = b_table_x + b_table_w + 40
        i_table_x = f_table_x + f_table_w + 40
        total_counter_width = i_table_x + i_table_w - b_table_x

        pygame.draw.rect(
            self.screen, FLOOR_COLOR, (40, 90, 1120, 580), border_radius=15
        )
        pygame.draw.rect(
            self.screen,
            COUNTER_COLOR,
            (b_table_x, 380, total_counter_width, 35),
            border_radius=5,
        )

        # Prep Tables
        pygame.draw.rect(
            self.screen, TABLE_COLOR, (b_table_x, 180, b_table_w, 50), border_radius=8
        )
        self.screen.blit(
            self.title_font.render("BURGER STATION", True, BG_COLOR),
            (b_table_x + 10, 215),
        )

        pygame.draw.rect(
            self.screen, TABLE_COLOR, (f_table_x, 180, f_table_w, 50), border_radius=8
        )
        self.screen.blit(
            self.title_font.render("FRY STATION", True, BG_COLOR), (f_table_x + 10, 215)
        )

        pygame.draw.rect(
            self.screen, TABLE_COLOR, (i_table_x, 180, i_table_w, 50), border_radius=8
        )
        self.screen.blit(
            self.title_font.render("DESSERT STATION", True, BG_COLOR),
            (i_table_x + 10, 215),
        )

        # --- 2. DRAW COOKS ---
        active_b = self.restaurant.burger_cook.count
        for i in range(self.num_burger_cooks):
            x = b_table_x + 35 + (i * 70)
            color = B_COOK_COLOR if i < active_b else IDLE_GRAY
            pygame.draw.circle(self.screen, BG_COLOR, (x, 130), 20)
            pygame.draw.circle(self.screen, color, (x, 130), 17)
            self.screen.blit(self.font.render("B", True, TEXT_COLOR), (x - 6, 120))

        active_f = self.restaurant.fries_cook.count
        for i in range(self.num_fries_cooks):
            x = f_table_x + 35 + (i * 70)
            color = F_COOK_COLOR if i < active_f else IDLE_GRAY
            pygame.draw.circle(self.screen, BG_COLOR, (x, 130), 20)
            pygame.draw.circle(self.screen, color, (x, 130), 17)
            self.screen.blit(self.font.render("F", True, TEXT_COLOR), (x - 5, 120))

        active_i = self.restaurant.ice_cream_cook.count
        for i in range(self.num_ice_cream_cooks):
            x = i_table_x + 35 + (i * 70)
            color = I_COOK_COLOR if i < active_i else IDLE_GRAY
            pygame.draw.circle(self.screen, BG_COLOR, (x, 130), 20)
            pygame.draw.circle(self.screen, color, (x, 130), 17)
            self.screen.blit(self.font.render("I", True, TEXT_COLOR), (x - 4, 120))

        # --- 3. DYNAMIC INVENTORY GRID ---
        for i in range(len(self.restaurant.burger_shelf.items)):
            max_cols = b_table_w // 25
            row, col = i // max_cols, i % max_cols
            bx, by = b_table_x + 10 + (col * 25), 185 - (row * 18)
            pygame.draw.rect(
                self.screen, (243, 156, 18), (bx, by, 18, 5), border_radius=3
            )
            pygame.draw.rect(self.screen, (121, 85, 72), (bx, by + 5, 18, 4))
            pygame.draw.rect(
                self.screen, (243, 156, 18), (bx, by + 9, 18, 4), border_radius=2
            )

        for i in range(len(self.restaurant.fries_shelf.items)):
            max_cols = f_table_w // 15
            row, col = i // max_cols, i % max_cols
            fx, fy = f_table_x + 10 + (col * 15), 185 - (row * 18)
            pygame.draw.rect(
                self.screen, (231, 76, 60), (fx, fy + 4, 10, 10), border_radius=2
            )
            pygame.draw.rect(self.screen, (241, 196, 15), (fx + 2, fy, 2, 6))
            pygame.draw.rect(self.screen, (241, 196, 15), (fx + 6, fy, 2, 6))

        for i in range(len(self.restaurant.ice_cream_shelf.items)):
            max_cols = i_table_w // 20
            row, col = i // max_cols, i % max_cols
            ix, iy = i_table_x + 10 + (col * 20), 185 - (row * 18)
            pygame.draw.polygon(
                self.screen,
                (210, 180, 140),
                [(ix, iy + 6), (ix + 12, iy + 6), (ix + 6, iy + 16)],
            )
            pygame.draw.circle(self.screen, (255, 250, 240), (ix + 6, iy + 4), 6)

        # --- 4. CASHIERS & CUSTOMERS ---
        active_c = self.restaurant.cashier.count
        c_spacing = total_counter_width // max(1, self.num_cashiers)

        for i in range(self.num_cashiers):
            x = b_table_x + (c_spacing // 2) + (i * c_spacing)
            color = CASHIER_COLOR if i < active_c else IDLE_GRAY
            pygame.draw.circle(self.screen, BG_COLOR, (x, 350), 20)
            pygame.draw.circle(self.screen, color, (x, 350), 17)
            self.screen.blit(self.font.render("$", True, TEXT_COLOR), (x - 4, 340))

            if i < active_c:
                pygame.draw.circle(self.screen, BG_COLOR, (x, 430), 18)
                pygame.draw.circle(self.screen, CUSTOMER_COLOR, (x, 430), 15)

        for i in range(len(self.restaurant.cashier.queue)):
            row, col = i // 25, i % 25
            pygame.draw.circle(
                self.screen, BG_COLOR, (120 + (col * 35), 480 + (row * 35)), 15
            )
            pygame.draw.circle(
                self.screen, CUSTOMER_COLOR, (120 + (col * 35), 480 + (row * 35)), 12
            )

        pickup_x = i_table_x + i_table_w + 40
        self.screen.blit(
            self.font.render("Pickup Area", True, TEXT_COLOR), (pickup_x, 350)
        )
        waiting_count = self.restaurant.customers_waiting_for_food

        for i in range(min(waiting_count, 20)):
            row, col = i // 4, i % 4
            pygame.draw.circle(
                self.screen, BG_COLOR, (pickup_x + (col * 35), 390 + (row * 35)), 15
            )
            pygame.draw.circle(
                self.screen,
                WAITING_FOOD_COLOR,
                (pickup_x + (col * 35), 390 + (row * 35)),
                12,
            )

        if waiting_count > 20:
            self.screen.blit(
                self.font.render(
                    f"+ {waiting_count - 20} waiting...", True, (231, 76, 60)
                ),
                (pickup_x, 390 + (5 * 35)),
            )

        # --- 5. EXPANDED UI DASHBOARD ---
        pygame.draw.rect(
            self.screen, UI_BAR_COLOR, (0, 0, 1200, 70)
        )  # Taller bar for two rows

        # Top Row: Standard Telemetry
        self.screen.blit(
            self.font.render(
                f"CLOCK: {self.env.now:.0f}s / {SIM_TIME}s", True, TEXT_COLOR
            ),
            (20, 10),
        )
        self.screen.blit(
            self.font.render(
                f"REVENUE: ${sum(self.stats['captured_revenue']):.2f}",
                True,
                (46, 204, 113),
            ),
            (250, 10),
        )

        waste_cost = (
            (len(self.stats["wasted_burgers"]) * COST_WASTED_BURGER)
            + (len(self.stats["wasted_fries"]) * COST_WASTED_FRIES)
            + (len(self.stats["wasted_ice_cream"]) * COST_WASTED_ICE_CREAM)
        )
        self.screen.blit(
            self.font.render(f"WASTE: ${waste_cost:.2f}", True, (241, 196, 15)),
            (500, 10),
        )
        self.screen.blit(
            self.font.render(
                f"WALK-OUTS: {len(self.stats['balked']) + len(self.stats['reneged'])}",
                True,
                (231, 76, 60),
            ),
            (750, 10),
        )

        # Bottom Row: Pending Orders Vision
        pending_text = (
            f"PENDING TICKETS ->  "
            f"BURGERS: {self.restaurant.pending_burgers}  |  "
            f"FRIES: {self.restaurant.pending_fries}  |  "
            f"ICE CREAM: {self.restaurant.pending_ice_cream}"
        )

        # Color it red if things are getting crazy (e.g. > 10 burgers waiting)
        warning_color = (
            (231, 76, 60) if self.restaurant.pending_burgers > 10 else (52, 152, 219)
        )
        self.screen.blit(self.font.render(pending_text, True, warning_color), (20, 40))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(10)
        elif self.render_mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
