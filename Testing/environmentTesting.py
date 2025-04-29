from Models.Enumerations.locationEnumeration import LocationEnum
from Models.enviromentSimulation import TaxiGymEnv

env = TaxiGymEnv()
loc, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    loc, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    print(f"Action: {action}, Taxi Location: {loc}, Reward: {reward}, Done: {done}, Info: {info}")

print("Total reward:", total_reward)