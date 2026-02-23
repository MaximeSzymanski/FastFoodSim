import os

# ==========================================
# 🎚️ DIFFICULTY TOGGLE
# Change this to "SIMPLE" or "NIGHTMARE"
# (Or set it via terminal: export DIFFICULTY="NIGHTMARE")
# ==========================================
DIFFICULTY = os.getenv("DIFFICULTY", "SIMPLE")

# --- SHARED CONSTANTS (Never Change) ---
SIM_TIME = 3600  # Run for 1 hour (in seconds)

CASHIER_MIN = 20.0
CASHIER_MAX = 120.0
CASHIER_MODE = 60.0

ICE_CREAM_TIME = 15.0
ICE_CREAM_BATCH_SIZE = 1
TARGET_ICE_CREAM_INV = 8
PRICE_ICE_CREAM = 3.00
WAGE_ICE_CREAM_COOK = 14.00

FRIES_BATCH_SIZE = 4
BURGER_BATCH_SIZE = 1
TARGET_FRIES_INV = 14
TARGET_BURGER_INV = 5

PRICE_BURGER = 8.00
PRICE_FRIES = 4.50
WAGE_CASHIER = 15.00
WAGE_BURGER_COOK = 18.00
WAGE_FRIES_COOK = 15.00


# ==========================================
# 🟢 SIMPLE MODE (The "Training Wheels")
# ==========================================
if DIFFICULTY == "SIMPLE":
    # Customer Flow
    ARRIVAL_AVG = 22.0
    MAX_QUEUE_LENGTH = 4
    MAX_WAIT_TOLERANCE = 180.0
    MAX_ORDER_WAITING_FOR_FOOD = 10

    # Process Times
    BURGER_MIN = 35.0
    BURGER_MAX = 60.0
    BURGER_MODE = 45.0
    FRIES_TIME = 150.0

    # Shelf Life
    ICE_CREAM_SHELF_LIFE = 60.0
    FRIES_SHELF_LIFE = 180.0
    BURGER_SHELF_LIFE = 300.0

    # Financial Penalties
    COST_WASTED_BURGER = 3.50
    COST_WASTED_FRIES = 1.00
    COST_WASTED_ICE_CREAM = 0.50

# ==========================================
# 🔴 NIGHTMARE MODE (The "Kitchen Hell")
# ==========================================
else:  # NIGHTMARE
    # Customer Flow
    ARRIVAL_AVG = 15.0  # Faster arrivals
    MAX_QUEUE_LENGTH = 3  # Lower patience
    MAX_WAIT_TOLERANCE = 120.0  # Lower patience
    MAX_ORDER_WAITING_FOR_FOOD = 6  # Lower patience

    # Process Times
    BURGER_MIN = 45.0
    BURGER_MAX = 75.0
    BURGER_MODE = 55.0  # Slower cooking
    FRIES_TIME = 180.0  # Slower cooking

    # Shelf Life
    ICE_CREAM_SHELF_LIFE = 30.0  # Melts instantly
    FRIES_SHELF_LIFE = 120.0  # Goes cold fast
    BURGER_SHELF_LIFE = 180.0  # Expires fast

    # Financial Penalties
    COST_WASTED_BURGER = 5.00  # Brutal margins
    COST_WASTED_FRIES = 2.00  # Brutal margins
    COST_WASTED_ICE_CREAM = 1.50  # Brutal margins

print(f"🔧 Loaded config with DIFFICULTY: {DIFFICULTY}")
