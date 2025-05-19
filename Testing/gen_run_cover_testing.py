import json
import os.path

from Models.environment_simulation import TaxiGymEnv
from Models.test_case_generator import TestCaseGenerator
from Models.test_case_runner import TestCaseRunner


env = TaxiGymEnv()

num_cases = 100
model_path = "../ppo_elements/best_model"
output_path = "../test_cases/"

# Generating Test Cases
print(f"Generating {num_cases} test cases...")
generator = TestCaseGenerator(model_path, env)
generator.generate(num_cases, output_path, "..\\Models")

# Running Test Cases
print("Running test cases...")
runner = TestCaseRunner(model_path, json.load(open(os.path.join(output_path, "generated_test_cases.json"), "r")))
results = runner.run_all()
success_rate = sum(1 for r in results if r) / len(results)
print(f"Test run complete: {len(results)} cases, {success_rate:.2%} success rate")
