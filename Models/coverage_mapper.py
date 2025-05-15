import ast
import json
import os
import coverage
from typing import List, Dict, Set


def build_function_map(root: str) -> dict[str, dict[str, list[tuple[int, int]]]]:
    func_map: Dict[str, List[tuple[str, int, int]]] = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if (not fn.endswith(".py")) or fn.startswith("."):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            funcs = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    start = node.lineno
                    end = max(
                        [n.lineno for n in ast.walk(node) if hasattr(n, "lineno")],
                        default=start
                    )
                    funcs.append((node.name, start, end))
            if len(funcs) > 0:
                func_map[path] = funcs
    return save_lines_map(func_map)


def save_lines_map(func_map: dict[str, list[tuple[str, int, int]]], path: str = "../test_cases/lines_map.json") -> dict[str, dict[str, list[tuple[int, int]]]]:
    lines = []
    line = {file_path.split("\\")[2]: {name: (start, end) for name, start, end in funcs} for file_path, funcs in func_map.items()}
    lines.append(line)
    lines = {k: v for d in lines for k, v in d.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=2)
    return json.load(open(path, "r"))


class CoverageMapper:
    """
    Tracks function-level coverage for each generated test case using coverage.py.
    """
    def __init__(self, source_root: str = "..\\Models"):
        self.cov = coverage.Coverage(data_file=None, omit=["*/site-packages/*"])
        self.func_map = build_function_map(source_root)
        self.root = source_root
        self.coverage_map: Dict[int, Set[str]] = {}
        self.current_test_id = -1

    def begin_test(self, test_id: int):
        """Call before running each test case."""
        self.current_test_id = test_id
        self.cov.start()

    def end_test(self):
        """Call after running each test case."""
        self.cov.stop()
        self.cov.save()

        data = self.cov.get_data()
        covered_funcs: Set[str] = set()
        for file_path, funcs in self.func_map.items():
            lines = data.lines(os.path.abspath(os.path.join(self.root, file_path))) or []
            for func_name in funcs:
                if any(funcs[func_name][0] <= ln <= funcs[func_name][1] for ln in lines) and func_name != "end_test":
                    covered_funcs.add(func_name)
        self.coverage_map[self.current_test_id] = covered_funcs
        self.cov.erase()

    def save_map(self, path: str = "../test_cases/coverage_map.json"):
        serializable = {
            tid: sorted(list(funcs))
            for tid, funcs in self.coverage_map.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
