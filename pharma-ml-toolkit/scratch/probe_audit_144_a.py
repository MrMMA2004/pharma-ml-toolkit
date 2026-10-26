"""Task 144-a probe A (static AST audit -- no Tk needed).

Checks, without launching anything:
  1. every self._start_job(...) call site's enclosing scope chain --
     a call from inside a NESTED function (a work/render closure) is
     flagged (busy-guard / worker-thread-Tk concerns);
  2. every ``self.ATTR.get()`` inside a nested function literally named
     ``work`` / ``work_single`` (the worker-thread bodies) -- the runtime
     half of the Tk-variable-from-worker check happens in probe B;
  3. lambdas defined inside for/while loops that close over the loop
     target (classic late binding).
"""

from __future__ import annotations

import ast
from pathlib import Path

PLAYGROUND = Path(__file__).resolve().parent.parent / "playground.py"
tree = ast.parse(PLAYGROUND.read_text(encoding="utf-8"))

# --- 1. _start_job call-site scopes ---------------------------------------- #
findings: list[str] = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "_start_job"):
        continue
    if not (isinstance(func.value, ast.Name) and func.value.id == "self"):
        continue
    # scope chain: (function name, lineno) from module down to the call
    chain: list[tuple[str, int]] = []
    scope: ast.AST | None = node
    # walk via parent map
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chain.append((cur.name, cur.lineno))
        cur = parents.get(id(cur))
    chain.reverse()
    nested = [name for name, _ in chain[2:]]  # [0]=method name, [1:]=nested
    if nested:
        findings.append(f"line {node.lineno}: _start_job inside nested {nested} of method {chain[0][0]!r}")
print("== 1. _start_job call sites inside nested closures ==")
print("\n".join(findings) if findings else "NONE (all 74+ call sites at method/if level)")

# --- 2. self.ATTR.get() inside work closures ------------------------------- #
print()
print("== 2. self.ATTR.get() calls inside nested work()/work_single() bodies ==")
parents: dict[int, ast.AST] = {}
for parent in ast.walk(tree):
    for child in ast.iter_child_nodes(parent):
        parents[id(child)] = parent


def enclosing_funcs(node: ast.AST) -> list[ast.FunctionDef]:
    out: list[ast.FunctionDef] = []
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.FunctionDef):
            out.append(cur)
        cur = parents.get(id(cur))
    return out[::-1]


getter_calls: list[str] = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "get"):
        continue
    if not (isinstance(func.value, ast.Attribute) and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"):
        continue
    attr = func.value.attr
    holders = enclosing_funcs(node)
    if holders and holders[-1].name in ("work", "work_single"):
        # exclude pd-like calls: self.ATTR.get() only -- attr names ending
        # in _var / _combo etc. are interesting; plain dict attrs too
        getter_calls.append(
            f"line {node.lineno}: self.{attr}.get() inside {holders[-1].name}() "
            f"of {holders[0].name}()"
        )
print("\n".join(getter_calls) if getter_calls else "NONE")

# --- 3. late-binding lambdas in loops --------------------------------------- #
print()
print("== 3. lambdas in for/while loops closing over the loop target ==")
loop_findings: list[str] = []
for node in ast.walk(tree):
    if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        continue
    target_names: set[str] = set()
    if isinstance(node, (ast.For, ast.AsyncFor)):
        for t in ast.walk(node.target):
            if isinstance(t, ast.Name):
                target_names.add(t.id)
    if not target_names:
        continue
    for sub in ast.walk(node):
        if isinstance(sub, ast.Lambda):
            used = {n.id for n in ast.walk(sub) if isinstance(n, ast.Name)}
            free = target_names & used
            # a defaulted parameter of the same name is the FIX, not a bug
            defaulted = {
                arg.arg for arg in sub.args.args + sub.args.kwonlyargs
                if arg.arg in free
            }
            if free - defaulted:
                loop_findings.append(
                    f"line {sub.lineno}: lambda in loop uses {sorted(free - defaulted)} "
                    f"without default-arg bind"
                )
print("\n".join(loop_findings) if loop_findings else "NONE")

print()
print("PROBE_A_STATIC_DONE")
