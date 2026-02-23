import random

import simpy

from src.config import *
from src.sim.restaurant import FastFoodRestaurant


class FoodItem:
    """Represents a food item created in the simulation.

    Args:
        creation_time (float): The exact simulation time the food item was created.
    """

    def __init__(self, creation_time):
        self.creation_time = creation_time


def fry_cook_loop(env, restaurant):
    """Continuously manages the fry cooking process to maintain target inventory.

    Args:
        env (simpy.Environment): The simulation environment.
        restaurant (FastFoodRestaurant): The restaurant instance containing the fry shelf.

    Yields:
        simpy.events.Timeout: The time delay required to cook the fries or wait.
    """
    while True:
        if len(restaurant.fries_shelf.items) < TARGET_FRIES_INV:
            yield env.timeout(FRIES_TIME)
            for _ in range(FRIES_BATCH_SIZE):
                restaurant.fries_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


def burger_cook_loop(env, restaurant):
    """Continuously manages the burger cooking process to maintain target inventory.

    Args:
        env (simpy.Environment): The simulation environment.
        restaurant (FastFoodRestaurant): The restaurant instance containing the burger shelf.

    Yields:
        simpy.events.Timeout: The time delay required to cook the burgers or wait.
    """
    while True:
        if len(restaurant.burger_shelf.items) < TARGET_BURGER_INV:
            burger_time = random.triangular(BURGER_MIN, BURGER_MAX, BURGER_MODE)
            yield env.timeout(burger_time)
            for _ in range(BURGER_BATCH_SIZE):
                restaurant.burger_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


def ice_cream_cook_loop(env, restaurant):
    """Continuously manages the ice cream pouring process to maintain target inventory.

    Args:
        env (simpy.Environment): The simulation environment.
        restaurant (FastFoodRestaurant): The restaurant instance containing the ice cream shelf.

    Yields:
        simpy.events.Timeout: The time delay required to pour the ice cream or wait.
    """
    while True:
        if len(restaurant.ice_cream_shelf.items) < TARGET_ICE_CREAM_INV:
            yield env.timeout(ICE_CREAM_TIME)
            for _ in range(ICE_CREAM_BATCH_SIZE):
                restaurant.ice_cream_shelf.put(FoodItem(env.now))
        else:
            yield env.timeout(5.0)


def inventory_manager(env, restaurant, stats):
    """Periodically checks food shelves and discards expired items.

    Args:
        env (simpy.Environment): The simulation environment.
        restaurant (FastFoodRestaurant): The restaurant instance containing the food shelves.
        stats (dict): The dictionary tracking simulation statistics, including waste.

    Yields:
        simpy.events.Timeout: The time interval between inventory checks.
    """
    while True:
        yield env.timeout(10.0)

        valid_burgers = []
        for item in restaurant.burger_shelf.items:
            if env.now - item.creation_time > BURGER_SHELF_LIFE:
                stats["wasted_burgers"].append(1)
            else:
                valid_burgers.append(item)
        restaurant.burger_shelf.items = valid_burgers

        valid_fries = []
        for item in restaurant.fries_shelf.items:
            if env.now - item.creation_time > FRIES_SHELF_LIFE:
                stats["wasted_fries"].append(1)
            else:
                valid_fries.append(item)
        restaurant.fries_shelf.items = valid_fries

        valid_ice_cream = []
        for item in restaurant.ice_cream_shelf.items:
            if env.now - item.creation_time > ICE_CREAM_SHELF_LIFE:
                stats["wasted_ice_cream"].append(1)
            else:
                valid_ice_cream.append(item)
        restaurant.ice_cream_shelf.items = valid_ice_cream


def grab_food_from_shelf(env, shelf):
    """Retrieves a food item from a specified shelf.

    Args:
        env (simpy.Environment): The simulation environment.
        shelf (simpy.Store): The specific inventory shelf to retrieve the item from.

    Yields:
        simpy.events.Get: An event representing the retrieval of a food item.
    """
    food = yield shelf.get()
    return food


def customer_journey(env, name, restaurant, stats):
    """Simulates the entire journey of a single customer through the restaurant.

    Args:
        env (simpy.Environment): The simulation environment.
        name (str): The unique identifier for the customer.
        restaurant (FastFoodRestaurant): The restaurant instance.
        stats (dict): The dictionary tracking simulation statistics.

    Yields:
        simpy.events.Event: Various simulation events representing waiting, ordering, and picking up food.
    """
    arrival_time = env.now

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

    if len(restaurant.cashier.queue) >= MAX_QUEUE_LENGTH:
        stats["balked"].append(1)
        stats["lost_revenue"].append(order_price)
        return
    if restaurant.customers_waiting_for_food >= MAX_ORDER_WAITING_FOR_FOOD:
        stats["balked"].append(1)
        stats["lost_revenue"].append(order_price)
        return

    with restaurant.cashier.request() as req:
        time_pct = env.now / SIM_TIME
        if 0.33 <= time_pct <= 0.66:
            current_patience = MAX_WAIT_TOLERANCE * 0.5
        else:
            current_patience = MAX_WAIT_TOLERANCE

        patience_timer = env.timeout(current_patience)
        results = yield req | patience_timer

        if req not in results:
            stats["reneged"].append(1)
            stats["lost_revenue"].append(order_price)
            return

        order_time = random.triangular(CASHIER_MIN, CASHIER_MAX, CASHIER_MODE)
        order_time += (num_burgers + num_fries + num_ice_cream) * 5.0
        yield env.timeout(order_time)

    restaurant.pending_burgers += num_burgers
    restaurant.pending_fries += num_fries
    restaurant.pending_ice_cream += num_ice_cream
    restaurant.customers_waiting_for_food += 1

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

    restaurant.pending_burgers -= num_burgers
    restaurant.pending_fries -= num_fries
    restaurant.pending_ice_cream -= num_ice_cream
    restaurant.customers_waiting_for_food -= 1

    departure_time = env.now
    stats["wait_times"].append(departure_time - arrival_time)
    stats["captured_revenue"].append(order_price)


def customer_arrivals(env, restaurant, stats):
    """Generates a continuous stream of arriving customers based on the time of day.

    Args:
        env (simpy.Environment): The simulation environment.
        restaurant (FastFoodRestaurant): The restaurant instance.
        stats (dict): The dictionary tracking simulation statistics.

    Yields:
        simpy.events.Timeout: The time delay before the next customer arrives.
    """
    count = 0
    while True:
        time_pct = env.now / SIM_TIME

        if 0.33 <= time_pct <= 0.66:
            current_arrival_rate = ARRIVAL_AVG * 0.4
        else:
            current_arrival_rate = ARRIVAL_AVG * 1.5

        yield env.timeout(random.expovariate(1.0 / current_arrival_rate))

        count += 1
        name = f"Customer {count}"
        env.process(customer_journey(env, name, restaurant, stats))
