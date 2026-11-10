"""A small, safe expression language for recipes (a subset of Python syntax).

Allowed: numbers, strings, names, arithmetic (+ - * / // % **), comparisons
(incl. `is None`), `and`/`or`/`not`, `&`/`|`/`~` on boolean arrays,
`a if cond else b`, function calls to whitelisted helpers, `x.key` / `x['key']` on
dicts (None if missing), lists, and f-strings (None formats as "–").
No attribute access on objects, no imports, no lambdas: a recipe can only combine
the signals and helpers the engine gives it.
"""

from __future__ import annotations

import ast
import operator as op
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
        ast.BitAnd: op.and_, ast.BitOr: op.or_}
_CMP = {ast.Lt: op.lt, ast.LtE: op.le, ast.Gt: op.gt, ast.GtE: op.ge, ast.Eq: op.eq,
        ast.NotEq: op.ne, ast.Is: op.is_, ast.IsNot: op.is_not,
        ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
_UNARY = {ast.USub: op.neg, ast.UAdd: op.pos, ast.Not: op.not_, ast.Invert: op.invert}


class ExprError(ValueError):
    pass


@lru_cache(maxsize=1024)
def parse(src: str) -> ast.Expression:
    try:
        return ast.parse(str(src).strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"bad expression {src!r}: {e.msg}") from None


class _Missing:
    """Lookup that returns None for unknown names (headline templates)."""


def evaluate(src: str | int | float | bool | None, names: Mapping[str, Any],
             missing_is_none: bool = False) -> Any:
    """Evaluate a recipe expression. Numbers/bools/None pass through unchanged."""
    if src is None or isinstance(src, bool | int | float):
        return src
    tree = parse(src)
    try:
        return _ev(tree.body, names, missing_is_none)
    except ExprError:
        raise
    except Exception as e:  # noqa: BLE001 - re-raise with the expression for context
        raise ExprError(f"in {src!r}: {type(e).__name__}: {e}") from e


def _ev(node: ast.AST, names: Mapping[str, Any], mn: bool) -> Any:
    ev = _ev
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in ("none", "None"):
            return None
        if node.id in ("true", "True"):
            return True
        if node.id in ("false", "False"):
            return False
        if mn:
            return None
        raise ExprError(f"unknown name {node.id!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](ev(node.left, names, mn), ev(node.right, names, mn))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](ev(node.operand, names, mn))
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            v = True
            for x in node.values:
                v = ev(x, names, mn)
                if not v:
                    return v
            return v
        v = False
        for x in node.values:
            v = ev(x, names, mn)
            if v:
                return v
        return v
    if isinstance(node, ast.Compare):
        left = ev(node.left, names, mn)
        result = True
        for o, c in zip(node.ops, node.comparators, strict=True):
            if type(o) not in _CMP:
                raise ExprError(f"operator {type(o).__name__} not allowed")
            right = ev(c, names, mn)
            r = _CMP[type(o)](left, right)
            result = r if result is True else result & r
            left = right
        return result
    if isinstance(node, ast.IfExp):
        return ev(node.body if ev(node.test, names, mn) else node.orelse, names, mn)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExprError("only helper functions can be called")
        fn = names.get(node.func.id)
        if not callable(fn):
            raise ExprError(f"unknown function {node.func.id!r}")
        args = [ev(a, names, mn) for a in node.args]
        kw = {k.arg: ev(k.value, names, mn) for k in node.keywords}
        return fn(*args, **kw)
    if isinstance(node, ast.Attribute):
        obj = ev(node.value, names, mn)
        return obj.get(node.attr) if isinstance(obj, Mapping) else None
    if isinstance(node, ast.Subscript):
        obj = ev(node.value, names, mn)
        key = ev(node.slice, names, mn)
        if obj is None:
            return None
        if isinstance(obj, Mapping):
            return obj.get(key)
        return obj[key]
    if isinstance(node, ast.List | ast.Tuple):
        return [ev(e, names, mn) for e in node.elts]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            else:
                val = ev(v.value, names, mn)
                spec = "".join(str(s.value) for s in v.format_spec.values) \
                    if v.format_spec else ""
                parts.append("–" if val is None else format(val, spec))
        return "".join(parts)
    raise ExprError(f"{type(node).__name__} not allowed in recipe expressions")
