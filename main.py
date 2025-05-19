from Models.agent_learning import AgentLearning
from Models.environment_simulation import TaxiGymEnv
from Models.prioritization import Prioritizer
from Models.test_case_generator import TestCaseGenerator
from Models.test_case_runner import TestCaseRunner

MODEL_PATH = "ppo_elements/best_model.zip"
TEST_CASES_PATH = "test_cases/generated_test_cases.json"


def generate_test_cases(model, env, count=100):
    print("Generating test cases...")
    generator = TestCaseGenerator(model, env)
    generator.generate(count)
    print(f"Saved {count} test cases.")


def prioritize_test_cases(test_cases):
    print("Prioritizing test cases...")
    prioritizer = Prioritizer(test_cases)
    prioritized = prioritizer.prioritize("./test_cases/")
    print("Prioritization complete.")
    return prioritized


def main():

    env = TaxiGymEnv()

    # Train or load the agent
    learning = AgentLearning(env)
    learning.load_or_train_model()

    # Generate test cases
    generate_test_cases(MODEL_PATH, env)

    # Prioritize test cases
    prioritized_cases = prioritize_test_cases(TEST_CASES_PATH)
    prioritized_cases = [item[1] for item in prioritized_cases]
    print("Prioritized test cases:", prioritized_cases)


if __name__ == "__main__":
    main()

