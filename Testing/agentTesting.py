from shimmy import GymV26CompatibilityV0
from stable_baselines3 import PPO

from Models.enviromentSimulation import TaxiGymEnv

env = GymV26CompatibilityV0(env = TaxiGymEnv())

#model = PPO("MlpPolicy", env, verbose=1)
#model.learn(total_timesteps=300_000)
#model.save("C:/Users/cerin/PycharmProjects/AutomatedTestCase/ppo_taxi_model_quick")

model = PPO.load("C:/Users/cerin/PycharmProjects/AutomatedTestCase/ppo_taxi_model_quick", env = env)
#model.learn(total_timesteps = 100_000)


ratio = 0.0
while ratio/1000 < 0.45:
    model.learn(total_timesteps=5_000_000)
    ratio = 0.0
    completed = 0
    incomplete = 0
    complete_less = 0

    for i in range(1000):
        loc, info = env.reset()
        done = False
        stopped = False
        total_reward = 0
        while not (done or stopped):
            action, _ = model.predict(loc)
            loc, reward, terminated, truncated, info = env.step(action)
            done = terminated
            stopped = truncated
            total_reward += reward
            #print(f"Action: {action}, Taxi Location: {loc}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            if info['steps'] < 150:
                complete_less += 1
            else:
                completed += 1
            ratio += 1.0
        else:
            incomplete += 1
        print(f"Total reward: {total_reward} -> NUMBER {i} -> Info: {info}")
    print("Completed: ", completed)
    print("Complete Small: ", complete_less)
    print("Truncated: ", incomplete)
    print("Ratio:", ratio/1000)
    model.save("C:/Users/cerin/PycharmProjects/AutomatedTestCase/ppo_taxi_model_quick")