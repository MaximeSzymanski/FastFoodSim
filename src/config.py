# --- SIMULATION SETTINGS ---
SIM_TIME = 3600  # Run for 1 hour (in seconds)

# --- PROCESS TIMES (in seconds) ---
ARRIVAL_AVG = 22.0  # HARD: A constant flood of customers (was 35.0)

CASHIER_MIN = 20.0
CASHIER_MAX = 120.0
CASHIER_MODE = 60.0

BURGER_MIN = 35.0
BURGER_MAX = 60.0
BURGER_MODE = 45.0  # HARD: Cooks take slightly longer per burger (was 35.0)

FRIES_TIME = 150.0  # HARD: Fries take longer to fry (was 120.0)

# --- BATCHING & INVENTORY HYPERPARAMETERS ---
FRIES_BATCH_SIZE = 4
BURGER_BATCH_SIZE = 1

TARGET_FRIES_INV = 14
TARGET_BURGER_INV = 5

FRIES_SHELF_LIFE = (
    180.0  # BRUTAL: Fries go cold and are tossed after 3 mins (was 300.0)
)
BURGER_SHELF_LIFE = 300.0  # BRUTAL: Burgers expire after 5 mins (was 600.0)

MAX_ORDER_WAITING_FOR_FOOD = (
    10  # HARD: Customers walk out if the pickup area looks crowded (was 15)
)

# --- CUSTOMER PATIENCE HYPERPARAMETERS ---
MAX_QUEUE_LENGTH = 4  # HARD: Customers balk if just 4 people are in line (was 5)
MAX_WAIT_TOLERANCE = 180.0  # HARD: Customers renege after waiting 3 mins (was 300.0)

# --- FINANCIAL SETTINGS ---
PRICE_BURGER = 8.00
PRICE_FRIES = 4.50

WAGE_CASHIER = 15.00
WAGE_BURGER_COOK = 18.00
WAGE_FRIES_COOK = 15.00

COST_WASTED_BURGER = 3.50  # PUNISHING: Waste hurts profit much more (was 1.50)
COST_WASTED_FRIES = 1.00  # PUNISHING: Waste hurts profit much more (was 0.50)
