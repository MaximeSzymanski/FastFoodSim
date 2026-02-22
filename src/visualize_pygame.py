import sys

import pygame
import simpy

from config import *
from processes import FoodItem, customer_arrivals, inventory_manager
from restaurant import FastFoodRestaurant

# --- COLORS ---
BG_COLOR = (30, 30, 40)
TEXT_COLOR = (240, 240, 240)
CUSTOMER_COLOR = (100, 200, 255)
IDLE_COLOR = (50, 200, 50)  # Green
BUSY_COLOR = (220, 50, 50)  # Red
BURGER_COLOR = (139, 69, 19)  # Brown
FRIES_COLOR = (255, 215, 0)  # Gold


def optuna_static_manager(env, restaurant):
    """A basic manager so the kitchen actually cooks while we watch."""
    while True:
        if len(restaurant.burger_shelf.items) < TARGET_BURGER_INV:
            if restaurant.burger_cook.count < restaurant.burger_cook.capacity:
                with restaurant.burger_cook.request() as req:
                    yield req
                    yield env.timeout(BURGER_MODE)
                    for _ in range(BURGER_BATCH_SIZE):
                        restaurant.burger_shelf.put(FoodItem(env.now))

        if len(restaurant.fries_shelf.items) < TARGET_FRIES_INV:
            if restaurant.fries_cook.count < restaurant.fries_cook.capacity:
                with restaurant.fries_cook.request() as req:
                    yield req
                    yield env.timeout(FRIES_TIME)
                    for _ in range(FRIES_BATCH_SIZE):
                        restaurant.fries_shelf.put(FoodItem(env.now))

        yield env.timeout(1.0)


def pygame_renderer(env, restaurant, stats, screen, clock, font):
    """
    This is a SimPy process that pauses the simulation to draw the Pygame window.
    It syncs the SimPy clock with the Pygame frame rate.
    """
    while True:
        # 1. Handle Window Closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # 2. Clear Screen
        screen.fill(BG_COLOR)

        # --- DRAW TOP BAR (STATS) ---
        revenue = sum(stats["captured_revenue"])
        wasted_b = len(stats["wasted_burgers"])
        wasted_f = len(stats["wasted_fries"])
        lost_cust = len(stats["balked"]) + len(stats["reneged"])

        time_text = font.render(f"Time: {env.now:.0f}s / {SIM_TIME}s", True, TEXT_COLOR)
        rev_text = font.render(f"Revenue: ${revenue:.2f}", True, (50, 250, 50))
        waste_text = font.render(
            f"Waste: {wasted_b}B / {wasted_f}F", True, (250, 150, 50)
        )
        lost_text = font.render(f"Lost Customers: {lost_cust}", True, BUSY_COLOR)

        screen.blit(time_text, (20, 20))
        screen.blit(rev_text, (250, 20))
        screen.blit(waste_text, (450, 20))
        screen.blit(lost_text, (650, 20))

        # --- DRAW CUSTOMER QUEUE ---
        q_text = font.render("Customer Line", True, TEXT_COLOR)
        screen.blit(q_text, (20, 100))

        queue_len = len(restaurant.cashier.queue)
        for i in range(queue_len):
            # Draw customers as blue circles
            x = 40 + (i * 30)
            if x > 350:  # Wrap line if it gets too long
                break
            pygame.draw.circle(screen, CUSTOMER_COLOR, (x, 150), 10)

        # --- DRAW CASHIERS ---
        c_text = font.render("Cashiers", True, TEXT_COLOR)
        screen.blit(c_text, (400, 100))

        active_cashiers = restaurant.cashier.count
        for i in range(restaurant.cashier.capacity):
            color = BUSY_COLOR if i < active_cashiers else IDLE_COLOR
            pygame.draw.rect(
                screen, color, (400 + (i * 50), 130, 40, 40), border_radius=5
            )

        # --- DRAW KITCHEN (COOKS) ---
        k_text = font.render("Kitchen Staff", True, TEXT_COLOR)
        screen.blit(k_text, (400, 250))

        # Burger Cooks
        b_cook_text = font.render("Burger:", True, TEXT_COLOR)
        screen.blit(b_cook_text, (320, 290))
        active_b_cooks = restaurant.burger_cook.count
        for i in range(restaurant.burger_cook.capacity):
            color = BUSY_COLOR if i < active_b_cooks else IDLE_COLOR
            pygame.draw.rect(
                screen, color, (400 + (i * 50), 280, 40, 40), border_radius=5
            )

        # Fries Cooks
        f_cook_text = font.render("Fries:", True, TEXT_COLOR)
        screen.blit(f_cook_text, (340, 350))
        active_f_cooks = restaurant.fries_cook.count
        for i in range(restaurant.fries_cook.capacity):
            color = BUSY_COLOR if i < active_f_cooks else IDLE_COLOR
            pygame.draw.rect(
                screen, color, (400 + (i * 50), 340, 40, 40), border_radius=5
            )

        # --- DRAW INVENTORY SHELVES ---
        inv_text = font.render("Ready Food Inventory", True, TEXT_COLOR)
        screen.blit(inv_text, (20, 250))

        # Burger Shelf
        b_inv = len(restaurant.burger_shelf.items)
        for i in range(b_inv):
            pygame.draw.rect(
                screen, BURGER_COLOR, (20 + (i * 25), 290, 20, 20), border_radius=3
            )

        # Fries Shelf
        f_inv = len(restaurant.fries_shelf.items)
        for i in range(f_inv):
            pygame.draw.rect(screen, FRIES_COLOR, (20 + (i * 15), 350, 10, 20))

        # 3. Update Screen and Control Speed
        pygame.display.flip()

        # This controls how fast the simulation looks.
        # 30 frames per second means 30 SimPy seconds pass in 1 real-world second.
        clock.tick(30)

        # Move SimPy time forward by 1 second
        yield env.timeout(1.0)


if __name__ == "__main__":
    # 1. Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((900, 450))
    pygame.display.set_caption("🍔 AI Fast Food Visualizer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20, bold=True)

    # 2. Initialize SimPy
    env = simpy.Environment()

    # 4 Cashiers, 2 Burger Cooks, 1 Fry Cook
    restaurant = FastFoodRestaurant(env, 4, 2, 1)

    stats = {
        "wait_times": [],
        "wasted_burgers": [],
        "wasted_fries": [],
        "balked": [],
        "reneged": [],
        "captured_revenue": [],
        "lost_revenue": [],
    }

    # 3. Boot up the Background Processes
    env.process(customer_arrivals(env, restaurant, stats))
    env.process(inventory_manager(env, restaurant, stats))
    env.process(optuna_static_manager(env, restaurant))

    # Boot up Pygame to draw everything
    env.process(pygame_renderer(env, restaurant, stats, screen, clock, font))

    # 4. Start!
    print("Launching Pygame window...")
    try:
        env.run(until=SIM_TIME)
        print("Shift Completed Successfully!")
        pygame.quit()
    except Exception as e:
        print(f"Simulation ended: {e}")
        pygame.quit()
