import random

import simpy
from pandas.core.apply import ResType

from config import *
from restaurant import FastFoodRestaurant


# --- 1. The Food Object ---
class FoodItem:
    def __init__(self, creation_time):
        self.creation_time = creation_time


# --- NEW: Active Waste Manager ---
def inventory_manager(env, restaurant, stats):
    """Continuously checks shelves and throws away expired food immediately."""
    while True:
        # Check the shelves every 10 seconds to align with the AI's step time
        yield env.timeout(10.0)

        # Check Burgers
        valid_burgers = []
        for item in restaurant.burger_shelf.items:
            if env.now - item.creation_time > BURGER_SHELF_LIFE:
                stats["wasted_burgers"].append(1)
            else:
                valid_burgers.append(item)
        restaurant.burger_shelf.items = valid_burgers

        # Check Fries
        valid_fries = []
        for item in restaurant.fries_shelf.items:
            if env.now - item.creation_time > FRIES_SHELF_LIFE:
                stats["wasted_fries"].append(1)
            else:
                valid_fries.append(item)
        restaurant.fries_shelf.items = valid_fries


# --- 3. Expiration Helper (Simplified) ---
def grab_food_from_shelf(env, shelf):
    """Simply grabs the food. The inventory_manager handles expiration."""
    food = yield shelf.get()
    return food


# --- 4. The Customer Journey ---
def customer_journey(
    env: simpy.Environment, name: str, restaurant: FastFoodRestaurant, stats
):
    arrival_time = env.now

    # Generate random order
    num_burgers = random.choices([0, 1, 2, 4], weights=[10, 60, 20, 10])[0]
    num_fries = random.choices([0, 1, 2, 3], weights=[20, 50, 20, 10])[0]
    if num_burgers == 0 and num_fries == 0:
        num_burgers = 1

    order_price = (num_burgers * PRICE_BURGER) + (num_fries * PRICE_FRIES)

    # 1. Balking
    if len(restaurant.cashier.queue) >= MAX_QUEUE_LENGTH:
        stats["balked"].append(1)
        stats["lost_revenue"].append(order_price)
        return

    # 2. Reneging (Waiting in line for Cashier)
    with restaurant.cashier.request() as req:
        patience_timer = env.timeout(MAX_WAIT_TOLERANCE)
        results = yield req | patience_timer

        if req not in results:
            stats["reneged"].append(1)
            stats["lost_revenue"].append(order_price)
            return

        order_time = random.triangular(CASHIER_MIN, CASHIER_MAX, CASHIER_MODE)
        order_time += (num_burgers + num_fries) * 5.0  # Add extra time for large orders
        yield env.timeout(order_time)

    restaurant.customers_waiting_for_food += 1
    # 3. Dynamic Pickup (Waiting at the pickup counter for food)
    food_tasks = []
    for _ in range(num_burgers):
        food_tasks.append(
            env.process(grab_food_from_shelf(env, restaurant.burger_shelf))
        )
    for _ in range(num_fries):
        food_tasks.append(
            env.process(grab_food_from_shelf(env, restaurant.fries_shelf))
        )

    if food_tasks:
        yield simpy.events.AllOf(env, food_tasks)

    restaurant.customers_waiting_for_food -= 1
    # 4. Success!
    departure_time = env.now
    stats["wait_times"].append(departure_time - arrival_time)
    stats["captured_revenue"].append(order_price)


def customer_arrivals(env, restaurant, stats):
    count = 0
    while True:
        yield env.timeout(random.expovariate(1.0 / ARRIVAL_AVG))
        count += 1
        name = f"Customer {count}"
        env.process(customer_journey(env, name, restaurant, stats))
