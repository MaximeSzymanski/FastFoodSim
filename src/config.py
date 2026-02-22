# --- SIMULATION SETTINGS ---
SIM_TIME = 3600 * 24  # Run for 1 hour (in seconds)

# --- PROCESS TIMES (in seconds) ---
ARRIVAL_AVG = 35.0  # Average time between customers

CASHIER_MIN = 20.0
CASHIER_MAX = 120.0
CASHIER_MODE = 60.0

BURGER_MIN = 30.0
BURGER_MAX = 50.0
BURGER_MODE = 35.0

FRIES_TIME = 120.0

# --- BATCHING & INVENTORY HYPERPARAMETERS ---
FRIES_BATCH_SIZE = 4  # A basket makes 4 portions of fries at once
BURGER_BATCH_SIZE = 1  # Cooks still make 1 burger at a time

TARGET_FRIES_INV = 14  # Stop frying if there are 8 portions on the shelf
TARGET_BURGER_INV = 5  # Stop grilling if there are 5 burgers on the shelf

FRIES_SHELF_LIFE = 300.0  # Throw away fries after 5 minutes (300s)
BURGER_SHELF_LIFE = 600.0  # Throw away burgers after 10 minutes (600s)

# --- CUSTOMER PATIENCE HYPERPARAMETERS ---
MAX_QUEUE_LENGTH = 5  # Leave immediately if 5 or more people are in the cashier line
MAX_WAIT_TOLERANCE = (
    300.0  # Leave the line if waiting for the cashier takes > 5 minutes (300s)
)

# --- FINANCIAL SETTINGS ---
PRICE_BURGER = 8.00
PRICE_FRIES = 4.50

WAGE_CASHIER = 15.00
WAGE_BURGER_COOK = 18.00
WAGE_FRIES_COOK = 15.00

COST_WASTED_BURGER = 1.50
COST_WASTED_FRIES = 0.50
