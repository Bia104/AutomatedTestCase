import json

from Models.environment_simulation import TaxiGymEnv
from Models.test_case_generator import TestCaseGenerator
from Models.test_case_runner import TestCaseRunner


env = TaxiGymEnv()

num_cases = 100
model_path = "../ppo_elements/models/best_model"
output_path = "../test_cases/generated_test_cases.json"

# Generating Test Cases
print(f"Generating {num_cases} test cases...")
generator = TestCaseGenerator(model_path, env, num_cases)
generator.generate()

# Running Test Cases
print("Running test cases...")
runner = TestCaseRunner(model_path, output_path)
results = runner.run_all()
success_rate = sum(1 for r in results if r) / len(results)
print(f"Test run complete: {len(results)} cases, {success_rate:.2%} success rate")
