import random

import simpy

from src.config import *
from src.sim.restaurant import FastFoodRestaurant


# --- 1. The Food Object ---
class FoodItem:
    def __init__(self, creation_time):
        self.creation_time = creation_time


# --- 2. The Autonomous Cook Processes ---
def fry_cook_loop(env, restaurant):
    """Continuously cooks fries if inventory is below target."""
    while True:
        if len(restaurant.fries_shelf.items) < TARGET_FRIES_INV:
            yield env.timeout(FRIES_TIME)
            for _ in range(FRIES_BATCH_SIZE):
                restaurant.fries_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


def burger_cook_loop(env, restaurant):
    """Continuously cooks burgers if inventory is below target."""
    while True:
        if len(restaurant.burger_shelf.items) < TARGET_BURGER_INV:
            burger_time = random.triangular(BURGER_MIN, BURGER_MAX, BURGER_MODE)
            yield env.timeout(burger_time)
            for _ in range(BURGER_BATCH_SIZE):
                restaurant.burger_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


def ice_cream_cook_loop(env, restaurant):
    """Continuously cooks ice cream if inventory is below target."""
    while True:
        if len(restaurant.ice_cream_shelf.items) < TARGET_ICE_CREAM_INV:
            yield env.timeout(ICE_CREAM_TIME)
            for _ in range(ICE_CREAM_BATCH_SIZE):
                restaurant.ice_cream_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


# --- 3. Active Waste Manager ---
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

        # Check Ice Cream
        valid_ice_cream = []
        for item in restaurant.ice_cream_shelf.items:
            if env.now - item.creation_time > ICE_CREAM_SHELF_LIFE:
                stats["wasted_ice_cream"].append(1)
            else:
                valid_ice_cream.append(item)
        restaurant.ice_cream_shelf.items = valid_ice_cream


# --- 4. Expiration Helper ---
def grab_food_from_shelf(env, shelf):
    """Simply grabs the food. The inventory_manager handles expiration."""
    food = yield shelf.get()
    return food


# --- 5. The Customer Journey ---
def customer_journey(
    env: simpy.Environment, name: str, restaurant: FastFoodRestaurant, stats
):
    arrival_time = env.now

    # Generate random order
    num_burgers = random.choices([0, 1, 2, 4], weights=[10, 60, 20, 10])[0]
    num_fries = random.choices([0, 1, 2, 3], weights=[20, 50, 20, 10])[0]
    num_ice_cream = random.choices([0, 1], weights=[70, 30])[0]

    if num_burgers == 0 and num_fries == 0 and num_ice_cream == 0:
        num_burgers = 1

    order_price = (
        (num_burgers * PRICE_BURGER)
        + (num_fries * PRICE_FRIES)
        + (num_ice_cream * PRICE_ICE_CREAM)
    )

    # 1. Balking
    if len(restaurant.cashier.queue) >= MAX_QUEUE_LENGTH:
        stats["balked"].append(1)
        stats["lost_revenue"].append(order_price)
        return
    if restaurant.customers_waiting_for_food >= MAX_ORDER_WAITING_FOR_FOOD:
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
        order_time += (
            num_burgers + num_fries + num_ice_cream
        ) * 5.0  # Add extra time for large orders
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
    for _ in range(num_ice_cream):
        food_tasks.append(
            env.process(grab_food_from_shelf(env, restaurant.ice_cream_shelf))
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
