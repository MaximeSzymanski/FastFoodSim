import random

import simpy

from config import *


# --- 1. The Food Object ---
class FoodItem:
    def __init__(self, creation_time):
        self.creation_time = creation_time


# --- 2. The Autonomous Cook Processes ---
def fry_cook_loop(env, restaurant):
    while True:
        if len(restaurant.fries_shelf.items) < TARGET_FRIES_INV:
            yield env.timeout(FRIES_TIME)
            for _ in range(FRIES_BATCH_SIZE):
                restaurant.fries_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


def burger_cook_loop(env, restaurant):
    while True:
        if len(restaurant.burger_shelf.items) < TARGET_BURGER_INV:
            burger_time = random.triangular(BURGER_MIN, BURGER_MAX, BURGER_MODE)
            yield env.timeout(burger_time)
            for _ in range(BURGER_BATCH_SIZE):
                restaurant.burger_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


# --- 3. Expiration Helper ---
def grab_food_from_shelf(env, shelf, shelf_life, waste_tracker_list):
    while True:
        food = yield shelf.get()
        age = env.now - food.creation_time
        if age <= shelf_life:
            return food
        else:
            waste_tracker_list.append(1)


# --- 4. The Customer Journey ---
def customer_journey(env, name, restaurant, stats):
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

    # 2. Reneging
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

    # 3. Dynamic Pickup (Gather all items simultaneously)
    food_tasks = []
    for _ in range(num_burgers):
        food_tasks.append(
            env.process(
                grab_food_from_shelf(
                    env,
                    restaurant.burger_shelf,
                    BURGER_SHELF_LIFE,
                    stats["wasted_burgers"],
                )
            )
        )
    for _ in range(num_fries):
        food_tasks.append(
            env.process(
                grab_food_from_shelf(
                    env, restaurant.fries_shelf, FRIES_SHELF_LIFE, stats["wasted_fries"]
                )
            )
        )

    if food_tasks:
        yield simpy.events.AllOf(env, food_tasks)

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
