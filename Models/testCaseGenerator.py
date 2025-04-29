from shimmy import GymV26CompatibilityV0
from stable_baselines3 import PPO
from Models.enviromentSimulation import TaxiGymEnv
from Models.Enumerations.actionEnumeration import ActionEnum

env = GymV26CompatibilityV0(env=TaxiGymEnv())

model = PPO.load("C:/Users/cerin/PycharmProjects/AutomatedTestCase/ppo_taxi_model", env=env)

test_cases = []
num_episodes = 10
test_id = 0

while len(test_cases) < num_episodes:
    loc, info = env.reset()
    done = False
    stopped = False
    actions = []
    start_pos = loc
    pass_loc = info['passenger_status']
    dest_loc = info['destination']

    while not (done or stopped):
        action_id, _ = model.predict(loc)
        action = ActionEnum(action_id).name
        actions.append(action)

        loc, reward, terminated, truncated, info = env.step(action_id)
        done = terminated
        stopped = truncated

    if done and info['steps'] < 100:
        test_case = {
            "TestID": test_id,
            "StartLocation": start_pos,
            "PassengerLocation": pass_loc,
            "DestinationLocation": dest_loc,
            "ActionSequence": actions,
            "Success": done
        }
        test_cases.append(test_case)
    test_id += 1

# Print or save test cases
for case in test_cases:
    print(f"--- Test {case['TestID']} ---")
    print(f"Start: {case['StartLocation']}")
    print(f"Passenger: {case['PassengerLocation']}")
    print(f"Destination: {case['DestinationLocation']}")
    print(f"Actions: {case['ActionSequence']}")
    print(f"Success: {case['Success']}")
    print()

import json

with open("generated_test_cases.json", "w") as f:
    json.dump(test_cases, f, indent=4)

    import json
    from Models.enviromentSimulation import TaxiGymEnv
    from Models.Enumerations.actionEnumeration import ActionEnum
    from shimmy import GymV26CompatibilityV0

    # Load the environment
    env = GymV26CompatibilityV0(env=TaxiGymEnv())

    # Load test cases
    with open("test_cases.json", "r") as f:
        test_cases = json.load(f)

    # Counters
    total_tests = len(test_cases)
    passed_tests = 0
    failed_tests = 0

    for idx, case in enumerate(test_cases):
        loc, info = env.reset()
        env.taxi_loc = tuple(case["StartLocation"])  # Force initial location manually

        done = False
        truncated = False

        for action_name in case["ActionSequence"]:
            action_id = ActionEnum[action_name].value
            loc, reward, done, truncated, info = env.step(action_id)

            if done or truncated:
                break

        if case["Success"] == done:
            print(f"[PASS] Test {idx} ✅")
            passed_tests += 1
        else:
            print(f"[FAIL] Test {idx} ❌")
            failed_tests += 1

    # Final Results
    print("\n--- Test Suite Results ---")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Pass Ratio: {passed_tests / total_tests:.4f}")




import json
from Models.enviromentSimulation import TaxiGymEnv
from Models.Enumerations.actionEnumeration import ActionEnum
from shimmy import GymV26CompatibilityV0

# Load the environment
env = GymV26CompatibilityV0(env=TaxiGymEnv())

# Load test cases
with open("test_cases.json", "r") as f:
    test_cases = json.load(f)

# Counters
total_tests = len(test_cases)
passed_tests = 0
failed_tests = 0

for idx, case in enumerate(test_cases):
    loc, info = env.reset()
    env.taxi_loc = tuple(case["StartLocation"])  # Force initial location manually

    done = False
    truncated = False

    for action_name in case["ActionSequence"]:
        action_id = ActionEnum[action_name].value
        loc, reward, done, truncated, info = env.step(action_id)

        if done or truncated:
            break

    if case["Success"] == done:
        print(f"[PASS] Test {idx}")
        passed_tests += 1
    else:
        print(f"[FAIL] Test {idx}")
        failed_tests += 1

# Final Results
print("\n--- Test Suite Results ---")
print(f"Total Tests: {total_tests}")
print(f"Passed: {passed_tests}")
print(f"Failed: {failed_tests}")
print(f"Pass Ratio: {passed_tests/total_tests:.4f}")


