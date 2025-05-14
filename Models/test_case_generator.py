import json
import numpy as np
from stable_baselines3 import PPO

from Models.Enumerations.actions import ActionEnum
from Models.coverage_mapper import CoverageMapper


class TestCaseGenerator:
    def __init__(self, agent, env, num_cases=1000):
        self.agent = PPO.load(agent)
        self.env = env
        self.num_cases = num_cases
        self.test_cases = []

    def generate(self, path: str ="../test_cases/generated_test_cases.json"):
        mapper = CoverageMapper()
        for i in range(self.num_cases):
            mapper.begin_test(i)
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
                "id": i,
                "start": start,
                "pickup": pickup,
                "destination": obs["destination"],
                "actions": [ActionEnum(a).name for a in steps],
                "success": done,
                "reward": total_reward,
                "length": len(steps)
            }

            self.test_cases.append(test_case)
            mapper.end_test()

        mapper.save_map()
        with open(path, "w") as f:
            json.dump(self.test_cases, f, indent=4)
