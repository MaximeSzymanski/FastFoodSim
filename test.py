import simpy


def alarm(env):
    yield env.timeout(5)
    print("Time to wake up ! ", env.now)


env = simpy.Environment()
env.process(alarm(env))
env.run()
