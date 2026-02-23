import os

DIFFICULTY = os.getenv(
    "DIFFICULTY", "SIMPLE"
)  # Sets the simulation difficulty level to SIMPLE or NIGHTMARE.

SIM_TIME = 3600  # Total duration of the simulation in seconds.

CASHIER_MIN = 20.0  # Minimum time in seconds to process a customer order.
CASHIER_MAX = 120.0  # Maximum time in seconds to process a customer order.
CASHIER_MODE = 60.0  # Most frequent time in seconds to process a customer order.

ICE_CREAM_TIME = 15.0  # Time in seconds required to pour one batch of ice cream.
ICE_CREAM_BATCH_SIZE = 1  # Number of ice cream units produced per cooking cycle.
TARGET_ICE_CREAM_INV = 8  # Desired inventory level for ice cream.
PRICE_ICE_CREAM = 3.00  # Selling price of one ice cream unit.
WAGE_ICE_CREAM_COOK = 14.00  # Hourly wage for the ice cream cook.

FRIES_BATCH_SIZE = 4  # Number of fry portions produced per cooking cycle.
BURGER_BATCH_SIZE = 1  # Number of burgers produced per cooking cycle.
TARGET_FRIES_INV = 14  # Desired inventory level for fries.
TARGET_BURGER_INV = 5  # Desired inventory level for burgers.

PRICE_BURGER = 8.00  # Selling price of one burger.
PRICE_FRIES = 4.50  # Selling price of one fry portion.
WAGE_CASHIER = 15.00  # Hourly wage for a cashier.
WAGE_BURGER_COOK = 18.00  # Hourly wage for a burger cook.
WAGE_FRIES_COOK = 15.00  # Hourly wage for a fry cook.

if DIFFICULTY == "SIMPLE":
    ARRIVAL_AVG = 22.0  # Average time in seconds between customer arrivals.
    MAX_QUEUE_LENGTH = 4  # Maximum number of people waiting before new customers balk.
    MAX_WAIT_TOLERANCE = (
        180.0  # Maximum time in seconds a customer will wait before reneging.
    )
    MAX_ORDER_WAITING_FOR_FOOD = (
        10  # Maximum number of pending orders before new customers balk.
    )

    BURGER_MIN = 35.0  # Minimum time in seconds to cook a burger.
    BURGER_MAX = 60.0  # Maximum time in seconds to cook a burger.
    BURGER_MODE = 45.0  # Most frequent time in seconds to cook a burger.
    FRIES_TIME = 150.0  # Time in seconds required to cook one batch of fries.

    ICE_CREAM_SHELF_LIFE = 60.0  # Time in seconds before ice cream melts and is wasted.
    FRIES_SHELF_LIFE = 180.0  # Time in seconds before fries go cold and are wasted.
    BURGER_SHELF_LIFE = 300.0  # Time in seconds before burgers expire and are wasted.

    COST_WASTED_BURGER = 3.50  # Financial penalty for a wasted burger.
    COST_WASTED_FRIES = 1.00  # Financial penalty for wasted fries.
    COST_WASTED_ICE_CREAM = 0.50  # Financial penalty for wasted ice cream.

else:
    ARRIVAL_AVG = 15.0  # Average time in seconds between customer arrivals.
    MAX_QUEUE_LENGTH = 3  # Maximum number of people waiting before new customers balk.
    MAX_WAIT_TOLERANCE = (
        120.0  # Maximum time in seconds a customer will wait before reneging.
    )
    MAX_ORDER_WAITING_FOR_FOOD = (
        6  # Maximum number of pending orders before new customers balk.
    )

    BURGER_MIN = 45.0  # Minimum time in seconds to cook a burger.
    BURGER_MAX = 75.0  # Maximum time in seconds to cook a burger.
    BURGER_MODE = 55.0  # Most frequent time in seconds to cook a burger.
    FRIES_TIME = 180.0  # Time in seconds required to cook one batch of fries.

    ICE_CREAM_SHELF_LIFE = 30.0  # Time in seconds before ice cream melts and is wasted.
    FRIES_SHELF_LIFE = 120.0  # Time in seconds before fries go cold and are wasted.
    BURGER_SHELF_LIFE = 180.0  # Time in seconds before burgers expire and are wasted.

    COST_WASTED_BURGER = 5.00  # Financial penalty for a wasted burger.
    COST_WASTED_FRIES = 2.00  # Financial penalty for wasted fries.
    COST_WASTED_ICE_CREAM = 1.50  # Financial penalty for wasted ice cream.

print(f"Loaded config with DIFFICULTY: {DIFFICULTY}")
