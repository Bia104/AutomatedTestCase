from stable_baselines3 import PPO
from shimmy import GymV26CompatibilityV0

from Models.Enumerations.locationEnumeration import LocationEnum
from Models.enviromentSimulation import TaxiGymEnv

env = GymV26CompatibilityV0(env = TaxiGymEnv(LocationEnum.Green, LocationEnum.Yellow))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

loc, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(loc)
    loc, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward

print("Total reward:", total_reward)
