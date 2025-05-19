from stable_baselines3 import PPO

from Models.Enumerations.actions import ActionEnum
from Models.Enumerations.locations import LocationEnum
from Models.environment_simulation import TaxiGymEnv
from Models.finite_state_machine import FSM


class TestCaseRunner:
    def __init__(self, model_path: str, test_cases):
        self.env = TaxiGymEnv()
        self.model = PPO.load(model_path, self.env)
        self.test_cases = test_cases

    def run_all(self):
        results = []
        for i, case in enumerate(self.test_cases):
            result = self.run_case(case)
            print(f"Test {i} | Success: {result}")
            results.append(result)
        return results

    def run_case(self, case):
        obs, _ = self.env.reset()
        self.env.taxi_loc = tuple(case["start"])
        self.env.pass_loc = LocationEnum(tuple(case["pickup"]))
        self.env.final_dest = LocationEnum(tuple(case["destination"]))
        self.env.fsm = FSM(self.env.final_dest, self.env.pass_loc)
        self.env.current_objective = self.env.pass_loc.value

        total_reward = 0
        success = False

        for action_str in case["actions"]:
            action = ActionEnum[action_str]
            obs, reward, done, truncated, info = self.env.step(action.value)
            total_reward += reward
            if done:
                success = True
                break
        return success == case["success"] and total_reward == case["reward"]
