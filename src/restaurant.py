import simpy


class FastFoodRestaurant:
    def __init__(self, env, num_cashiers):
        self.env = env

        # Human resources
        self.cashier = simpy.Resource(env, num_cashiers)

        # Physical inventory shelves
        self.burger_shelf = simpy.Store(env)
        self.fries_shelf = simpy.Store(env)
