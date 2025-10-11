import ast
from typing import Any


def get_dtype(value: Any) -> str:
    dtype = {
        int: "i32",
        float: "f32",
    }.get(type(value))
    if dtype is None:
        raise ValueError(f"Unsupported type: {type(value)}")
    return dtype


class Value:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Scalar(Value):
    def __init__(self, name: str, dtype: str) -> None:
        super().__init__(name)
        self.dtype = dtype

    def __repr__(self) -> str:
        return self.dtype


class Writer:
    def __init__(self) -> None:
        self.ssa = -1
        self.lines = []
        self.indent = 0

    def __repr__(self) -> str:
        return "\n".join(self.lines)

    def append(self, line: str) -> None:
        self.lines.append("  " * self.indent + line)

    def scalar(self, dtype: str) -> Scalar:
        self.ssa += 1
        return Scalar(f"%{self.ssa}", dtype)

    def constant(self, value: Any) -> Scalar:
        res = self.scalar(get_dtype(value))
        self.append(f"{res.name} = arith.constant {value} : {res}")
        return res

    def binary(self, op: str, left: Value, right: Value) -> Value:
        assert left.dtype == right.dtype
        op = f"arith.{op}{left.dtype[0]}"
        res = self.scalar(left.dtype)
        self.append(f"{res.name} = {op} {left.name}, {right.name} : {left}")
        return res


class Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.writer = Writer()
        self.values = dict()

    def __repr__(self):
        return repr(self.writer)

    def get_value(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Name):
            return self.values[node.id]
        return self.values[node]

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        self.values[node.targets[0].id] = self.values[node.value]

    def visit_Constant(self, node: ast.Constant) -> None:
        self.generic_visit(node)
        self.values[node] = self.writer.constant(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.generic_visit(node)
        opname = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
        }[type(node.op)]
        left = self.get_value(node.left)
        right = self.get_value(node.right)
        result = self.writer.binary(opname, left, right)
        self.values[node] = result
