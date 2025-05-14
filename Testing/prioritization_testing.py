from Models.prioritization import TestCasePrioritizer

prioritizer = TestCasePrioritizer("../test_cases/generated_test_cases.json")
for test_case in prioritizer.prioritize():
    print(test_case)