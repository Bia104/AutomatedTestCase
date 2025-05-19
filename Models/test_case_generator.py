import json
import os.path

import numpy as np
from stable_baselines3 import PPO

from Models.Enumerations.actions import ActionEnum
from Models.coverage_mapper import CoverageMapper


class TestCaseGenerator:
    def __init__(self, agent, env):
        self.agent = PPO.load(agent)
        self.env = env
        self.test_cases = []

    def generate(self, num_cases:int = 1000, path: str ="./test_cases/", root: str = ".\\Models"):
        mapper = CoverageMapper(root, path)
        for i in range(num_cases):
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

        mapper.save_map(os.path.join(path, "coverage_map.json"))
        with open(os.path.join(path, "generated_test_cases.json"), "w") as f:
            json.dump(self.test_cases, f, indent=4)
