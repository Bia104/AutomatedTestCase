from stable_baselines3 import PPO

from Models.enviromentSimulation import TaxiGymEnv

env = TaxiGymEnv()
model = PPO.load("../ppo_elements/models/best_model", env = env)

ratio = 0.0
avg_steps = 0

for i in range(1000):
    obs, info = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if terminated:
            ratio += 1
        #print(f"Action: {action}, Taxi Location: {obs['taxi_loc']}, Reward: {reward}, Done: {done}, Info: {info}")
    avg_steps += info['steps']
    print(f"Total reward: {total_reward} -> NUMBER {i} -> Info: {info}")
print(f"\nRatio: {ratio/10}% \nAverage Steps: {int(avg_steps/1000)}")