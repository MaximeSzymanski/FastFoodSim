import simpy


class FastFoodRestaurant:
    def __init__(self, env: simpy.Environment, num_cashiers: int, num_cooks: int):
        self.env = env
        # These are the limited resources in your restaurant
        self.cashier: simpy.Resource = simpy.Resource(env, num_cashiers)
        self.cook: simpy.Resource = simpy.Resource(env, num_cooks)
