import os
import sys
import warnings

# Hide the deprecation warning and the Pygame welcome message
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame
import simpy

from config import *
from sim.processes import FoodItem, customer_arrivals, inventory_manager
from sim.restaurant import FastFoodRestaurant

# --- COLORS (RGB Tuples) ---
BG_COLOR = (30, 30, 35)  # Dark background
FLOOR_COLOR = (50, 50, 55)  # Kitchen/Lobby floor
TEXT_COLOR = (240, 240, 240)

# Actor Colors
CUSTOMER_COLOR = (100, 200, 255)  # Light Blue
CASHIER_COLOR = (180, 80, 255)  # Purple
B_COOK_COLOR = (255, 80, 80)  # Red
F_COOK_COLOR = (255, 160, 50)  # Orange
IDLE_GRAY = (80, 80, 80)  # Gray for idle employees
WAITING_FOOD_COLOR = (255, 255, 100)  # Yellow

# Object Colors
COUNTER_COLOR = (120, 90, 70)  # Wood
BURGER_COLOR = (139, 69, 19)  # Dark Brown
FRIES_COLOR = (255, 215, 0)  # Gold


def cook_burger_task(env, restaurant):
    """Background task for a cook to actually make the burger."""
    with restaurant.burger_cook.request() as req:
        yield req
        yield env.timeout(BURGER_MODE)
        for _ in range(BURGER_BATCH_SIZE):
            restaurant.burger_shelf.put(FoodItem(env.now))


def cook_fries_task(env, restaurant):
    """Background task for a cook to actually make the fries."""
    with restaurant.fries_cook.request() as req:
        yield req
        yield env.timeout(FRIES_TIME)
        for _ in range(FRIES_BATCH_SIZE):
            restaurant.fries_shelf.put(FoodItem(env.now))


def optuna_static_manager(env, restaurant):
    """The manager now delegates tasks and immediately moves on to enable parallel cooking."""
    while True:
        # 1. Check Burgers
        if len(restaurant.burger_shelf.items) < TARGET_BURGER_INV:
            # Check how many cooks are currently busy OR waiting for a task
            active_b_tasks = restaurant.burger_cook.count + len(
                restaurant.burger_cook.queue
            )

            # If we have an idle cook, assign them a task in the background!
            if active_b_tasks < restaurant.burger_cook.capacity:
                env.process(cook_burger_task(env, restaurant))

        # 2. Check Fries
        if len(restaurant.fries_shelf.items) < TARGET_FRIES_INV:
            active_f_tasks = restaurant.fries_cook.count + len(
                restaurant.fries_cook.queue
            )

            # If we have an idle cook, assign them a task in the background!
            if active_f_tasks < restaurant.fries_cook.capacity:
                env.process(cook_fries_task(env, restaurant))

        # Wait just 1 second before scanning the kitchen again
        yield env.timeout(1.0)


def topdown_renderer(env, restaurant, stats, screen, clock, font):
    """Draws the top-down 2D map of the restaurant."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(BG_COLOR)

        # Draw the main building floor
        pygame.draw.rect(screen, FLOOR_COLOR, (50, 50, 700, 500), border_radius=10)

        # --- DRAW STATIC ARCHITECTURE ---
        # The Main Order Counter
        pygame.draw.rect(screen, COUNTER_COLOR, (100, 350, 600, 30))

        # Kitchen Prep Tables
        pygame.draw.rect(screen, COUNTER_COLOR, (150, 150, 200, 40))  # Burger Prep
        pygame.draw.rect(screen, COUNTER_COLOR, (450, 150, 200, 40))  # Fries Prep

        # --- DRAW ACTORS & INVENTORY ---

        # 1. Cooks (Top of the screen)
        active_b = restaurant.burger_cook.count
        for i in range(restaurant.burger_cook.capacity):
            x, y = 200 + (i * 80), 100
            color = B_COOK_COLOR if i < active_b else IDLE_GRAY
            pygame.draw.circle(screen, color, (x, y), 18)
            screen.blit(font.render("B", True, TEXT_COLOR), (x - 6, y - 10))

        active_f = restaurant.fries_cook.count
        for i in range(restaurant.fries_cook.capacity):
            x, y = 500 + (i * 80), 100
            color = F_COOK_COLOR if i < active_f else IDLE_GRAY
            pygame.draw.circle(screen, color, (x, y), 18)
            screen.blit(font.render("F", True, TEXT_COLOR), (x - 5, y - 10))

        # 2. Food Inventory (On the prep tables)
        b_inv = len(restaurant.burger_shelf.items)
        for i in range(b_inv):
            pygame.draw.rect(
                screen, BURGER_COLOR, (160 + (i * 20), 160, 15, 15), border_radius=3
            )

        f_inv = len(restaurant.fries_shelf.items)
        for i in range(f_inv):
            pygame.draw.rect(screen, FRIES_COLOR, (460 + (i * 12), 160, 8, 15))

        # 3. Cashiers (Behind the counter)
        active_c = restaurant.cashier.count
        for i in range(restaurant.cashier.capacity):
            x, y = 200 + (i * 120), 320
            color = CASHIER_COLOR if i < active_c else IDLE_GRAY
            pygame.draw.circle(screen, color, (x, y), 15)
            screen.blit(font.render("$", True, TEXT_COLOR), (x - 4, y - 10))

            # 4. Active Ordering Customers (In front of the counter)
            if i < active_c:
                pygame.draw.circle(screen, CUSTOMER_COLOR, (x, 400), 15)

        # 5. Customer Line Queue (Lobby area)
        queue_len = len(restaurant.cashier.queue)
        for i in range(queue_len):
            row = i // 15
            col = i % 15
            x = 120 + (col * 35)
            y = 450 + (row * 35)
            pygame.draw.circle(screen, CUSTOMER_COLOR, (x, y), 12)

        # 6. Waiting for Food (At the pickup counter)
        pickup_text = font.render("Pickup Area", True, TEXT_COLOR)
        screen.blit(pickup_text, (600, 320))

        waiting_count = restaurant.customers_waiting_for_food
        MAX_VISUAL_PICKUP = 16

        for i in range(min(waiting_count, MAX_VISUAL_PICKUP)):
            row = i // 4
            col = i % 4
            x = 600 + (col * 30)
            y = 360 + (row * 30)
            pygame.draw.circle(screen, WAITING_FOOD_COLOR, (x, y), 12)

        # Draw text if the pickup area overflows
        if waiting_count > MAX_VISUAL_PICKUP:
            overflow = waiting_count - MAX_VISUAL_PICKUP
            overflow_text = font.render(
                f"+ {overflow} more waiting...", True, (255, 100, 100)
            )
            screen.blit(overflow_text, (600, 360 + (4 * 30)))

        # --- DRAW LIVE FINANCIAL STATS ---
        time_text = font.render(
            f"Simulation Clock: {env.now:.0f}s / {SIM_TIME}s", True, TEXT_COLOR
        )
        rev_text = font.render(
            f"Captured Revenue: ${sum(stats['captured_revenue']):.2f}",
            True,
            (100, 255, 100),
        )
        lost_text = font.render(
            f"Customers Walked Out: {len(stats['balked']) + len(stats['reneged'])}",
            True,
            (255, 100, 100),
        )

        screen.blit(time_text, (20, 10))
        screen.blit(rev_text, (300, 10))
        screen.blit(lost_text, (550, 10))

        # Push everything to the screen and control the framerate
        pygame.display.flip()
        clock.tick(60)

        # Advance SimPy time!
        yield env.timeout(1.0)


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("🍔 Top-Down Restaurant Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16, bold=True)

    env = simpy.Environment()

    # I set this back to 4 Cashiers, 2 Burger Cooks, and 1 Fry Cook to match standard Optuna.
    # You can change it to 4, 4, 1 if you want a massive kitchen!
    restaurant = FastFoodRestaurant(env, 4, 4, 1)

    stats = {
        "wait_times": [],
        "wasted_burgers": [],
        "wasted_fries": [],
        "balked": [],
        "reneged": [],
        "captured_revenue": [],
        "lost_revenue": [],
    }

    env.process(customer_arrivals(env, restaurant, stats))
    env.process(inventory_manager(env, restaurant, stats))
    env.process(optuna_static_manager(env, restaurant))

    # Launch the visualizer process
    env.process(topdown_renderer(env, restaurant, stats, screen, clock, font))

    print("Launching Pygame Top-Down View...")
    try:
        env.run(until=SIM_TIME)
        pygame.quit()
    except Exception as e:
        print(f"Simulation ended: {e}")
        pygame.quit()
