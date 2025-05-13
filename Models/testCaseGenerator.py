import json
import numpy as np
from stable_baselines3 import PPO

from Models.Enumerations.actionEnumeration import ActionEnum


class TestCaseGenerator:
    def __init__(self, agent, env, num_cases=1000):
        self.agent = PPO.load(agent)
        self.env = env
        self.num_cases = num_cases
        self.test_cases = []

    def generate(self, path="../test_cases/generated_test_cases.json"):
        for _ in range(self.num_cases):
            obs, _ = self.env.reset()
            steps = []
            done = False
            truncated = False
            total_reward = 0

            start = obs["taxi_loc"]
            pickup = obs["pickup_location"]

            while not (done or truncated):
                # Early break to generate partial paths
                if np.random.random() < 0.1:  # 10% chance to break early
                    break
                action, _ = self.agent.predict(obs)
                steps.append(action)
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward

            test_case = {
                "start": start,
                "pickup": pickup,
                "destination": obs["destination"],
                "actions": [ActionEnum(a).name for a in steps],
                "success": done,
                "reward": total_reward,
                "length": len(steps)
            }

            self.test_cases.append(test_case)
        # Saving the Test Cases
        with open(path, "w") as f:
            json.dump(self.test_cases, f, indent=4)
