import json
import subprocess
import difflib
import ast
import os


def is_git_repo() -> bool:
    result = subprocess.run([
        "git", "rev-parse", "--is-inside-work-tree"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip() == "true"


def get_modified_functions(diff_text: str) -> set[str]:
    lines = diff_text.splitlines()
    modified_funcs = set()

    for line in lines:
        if line.startswith("@@"):
            context = line.split("@@")[-1].strip()
            if "def " in context:
                func_name = context.split("def ")[-1].split("(")[0]
                modified_funcs.add(func_name)
    return modified_funcs

def get_diff(path: str, base_commit="HEAD") -> str:
    try:
        result = subprocess.run(
            ["git", "diff", base_commit, "--function-context", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Git diff failed:", e.stderr)
        return ""


class TestCasePrioritizer:
    def __init__(self, test_cases_path: str):
        self.test_cases = json.load(open(test_cases_path, "r"))

    def prioritize(self) -> list[dict]:
        if not is_git_repo():
            print("Not a git repository.")
            return []
        else:
            diff = get_diff("..")
            changed_functions = get_modified_functions(diff)
            original_functions = json.load(open("../test_cases/coverage_map.json", "r"))

            return prioritize_tests(self.test_cases, changed_functions, original_functions)

def prioritize_tests(test_cases: list[dict], modified: set[str],
                     original: dict[int, set[str]]) -> list[dict]:
    scored = []
    for tc in test_cases:
        funcs = original.get(tc["id"], set())
        s = len(funcs & modified)
        scored.append((s, tc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [tc for score, tc in scored]
