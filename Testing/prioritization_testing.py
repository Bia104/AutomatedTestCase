from Models.prioritization import Prioritizer

prioritizer = Prioritizer("../test_cases/generated_test_cases.json")
for test_case in prioritizer.prioritize():
    print(test_case)