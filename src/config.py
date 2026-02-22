# --- SIMULATION SETTINGS ---
SIM_TIME = 3600 * 24 * 30  # Run for 1 hour (in seconds)

# --- EMPLOYEES / RESOURCES ---
NUM_CASHIERS = 2
NUM_COOKS = 4

# --- PROCESS TIMES (in seconds) ---
# Arrivals
ARRIVAL_AVG = 45.0  # On average, a new customer every 45 seconds

# Cashier (Triangular: min, max, mode/avg)
CASHIER_MIN = 20.0
CASHIER_MAX = 120.0
CASHIER_MODE = 60.0

# Burger (Triangular: min, max, mode/avg)
BURGER_MIN = 30.0
BURGER_MAX = 50.0
BURGER_MODE = 35.0

# Fries (Constant time)
# FRIES_TIME = 120.0
# Note: If you meant a uniform range, you would use:
FRIES_MIN = 110.0
FRIES_MAX = 130.0

# Financial settings
ORDER_VALUE = 10.0  # Average revenue per order ($)
WAGE_CASHIER = 15.0  # $ per hour
WAGE_COOK = 17.0  # $ per hour
