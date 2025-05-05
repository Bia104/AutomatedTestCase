from Models.enviromentSimulation import TaxiGymEnv

env = TaxiGymEnv()
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    print(f"Action: {action}, Observed: {obs}, Reward: {reward}, Done: {done}, Info: {info}")

print("Total reward:", total_reward)