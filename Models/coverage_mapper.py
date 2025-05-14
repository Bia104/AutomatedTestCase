import json
import os

import coverage
import ast
import inspect
from typing import List, Dict, Set


def build_function_map(root: str) -> Dict[str, List[tuple[str, int, int]]]:
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
            func_map[path] = funcs
    return func_map


class CoverageMapper:
    """
    Tracks function-level coverage for each generated test case using coverage.py.
    """
    def __init__(self, source_root: str = "..\\Models"):
        self.cov = coverage.Coverage(data_file=None, omit=["*/site-packages/*"])
        self.func_map = build_function_map(source_root)
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
            lines = data.lines(os.path.abspath(file_path)) or []
            for func_name, start, end in funcs:
                if any(start <= ln <= end for ln in lines):
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
