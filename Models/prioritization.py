import json
import os.path
import subprocess


def is_git_repo() -> bool:
    result = subprocess.run([
        "git", "rev-parse", "--is-inside-work-tree"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip() == "true"


def get_modified_functions(diff_text: str, lines_map_path: str) -> set[str]:
    modified_funcs = set()
    lines_map = json.load(open(lines_map_path, "r"))
    current_class = ""
    changed = False

    for line in diff_text.splitlines():
        if line.startswith("+++"):
            if line.endswith(".py") and not (line.__contains__("Testing") or line.__contains__("main")):
                current_class = line.split("+++")[1].split("/")[2].strip()
                changed = True
            else:
                changed = False
        if line.startswith("@@") and changed:
            context = line.split("@@")[1].strip()
            if "+" in context:
                plus_part = context.split("+")[1].split(" ")[0]
                new_start, new_count = map(int, plus_part.split(",")) if "," in plus_part else (int(plus_part), 1)
                for l in lines_map[current_class]:
                    if (lines_map[current_class][l][0] <= int(new_start) <= lines_map[current_class][l][1]
                            or lines_map[current_class][l][0] <= int(new_start) + int(new_count) - 1 <= lines_map[current_class][l][1] + 1):
                        modified_funcs.add(l)
    return modified_funcs

def get_diff(path: str, base_commit="HEAD") -> str:
    try:
        result = subprocess.run(
            ["git", "diff", base_commit, "-U0", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Git diff failed:", e.stderr)
        return ""


class Prioritizer:
    def __init__(self, test_cases_path: str, root:str = "./Models"):
        self.test_cases = json.load(open(test_cases_path, "r"))
        self.root = root

    def prioritize(self, save_path:str = "./test_cases/") -> list[dict]:
        if not is_git_repo():
            print("Not a git repository.")
            return []
        else:
            diff = get_diff(self.root)
            changed_functions = get_modified_functions(diff, os.path.join(save_path, "lines_map.json"))
            original_functions = json.load(open(os.path.join(save_path, "coverage_map.json"), "r"))

            combined = (prioritize_tests(self.test_cases, changed_functions, original_functions) +
                        prioritize_classes(changed_functions, os.path.join(save_path, "lines_map.json"),
                                            os.path.join(save_path, "classes_coverage.json")))
            combined.sort(reverse=True, key=lambda x: x[0])
            return combined

def prioritize_tests(test_cases: list[dict], modified: set[str],
                     original: dict[str, set[str]]) -> list[dict]:
    scored = []
    for tc in test_cases:
        funcs = set(original.get(f"{tc["id"]}", set()))
        s = len(funcs.intersection(modified))
        scored.append((s, tc))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored

def prioritize_classes(changed:set[str], lines_path: str, classes_path: str) -> list[dict]:
    with open(classes_path, "r", encoding="utf-8") as f:
        classes_coverage: dict[str, set[str]] = json.load(f)

    with open(lines_path, "r", encoding="utf-8") as f:
        lines_map: dict[str, list[str]] = json.load(f)


    method_to_class: dict[str, str] = {}
    for model_cls, methods in lines_map.items():
        for m in methods:
            method_to_class[m] = model_cls

    changed_model_classes = {}
    for m in changed:
        if m in method_to_class:
            try:
                changed_model_classes[method_to_class[m]] += 1
            except KeyError:
                changed_model_classes.update({method_to_class[m]: 1})

    scored = []
    score = 0
    for test_cls, covered_models in classes_coverage.items():
        for model in covered_models:
            if model in changed_model_classes:
                score += changed_model_classes[model]
        scored.append((score, test_cls))

    scored.sort(key=lambda x: -x[0])

    return scored