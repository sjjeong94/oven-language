import ast
from typing import Any, List, Optional


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


class Vector(Scalar):
    def __init__(self, name: str, dtype: str, shape: List[int]) -> None:
        super().__init__(name, dtype)
        self.shape: List[int] = shape

    def __repr__(self) -> str:
        shape = "x".join(str(v) for v in self.shape)
        return f"vector<{shape}x{self.dtype}>"


class Pointer(Value):
    def __init__(self, name: str, space: Optional[int] = None) -> None:
        super().__init__(name)
        self.space = space

    def __repr__(self) -> str:
        space = "" if self.space is None else f"<{self.space}>"
        return f"!llvm.ptr{space}"


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

    def vector(self, dtype: str, shape: List[int]) -> Vector:
        self.ssa += 1
        return Vector(f"%{self.ssa}", dtype, shape)

    def pointer(self, space: Optional[int] = None) -> Pointer:
        self.ssa += 1
        return Pointer(f"%{self.ssa}", space)

    def constant(self, value: Any) -> Scalar:
        res = self.scalar(get_dtype(value))
        self.append(f"{res.name} = arith.constant {value} : {res}")
        return res

    def unary(self, op: str, operand: Value) -> Value:
        res = self.scalar(operand.dtype)
        self.append(f"{res.name} = {op} {operand.name} : {operand}")
        return res

    def binary(self, op: str, left: Value, right: Value) -> Value:
        assert left.dtype == right.dtype
        op = f"arith.{op}{left.dtype[0]}"
        res = self.scalar(left.dtype)
        self.append(f"{res.name} = {op} {left.name}, {right.name} : {left}")
        return res

    def ret(self, values: List[Value]) -> None:
        names = ", ".join(v.name for v in values)
        types = ", ".join(repr(v) for v in values)
        self.append(f"return {names} : {types}")

    def get_op(self, op: str) -> Scalar:
        if op in {"get_bdim_x", "get_bdim_y", "get_bdim_z"}:
            opname = "nvvm.read.ptx.sreg.ntid."
        elif op in {"get_bid_x", "get_bid_y", "get_bid_z"}:
            opname = "nvvm.read.ptx.sreg.ctaid."
        elif op in {"get_tid_x", "get_tid_y", "get_tid_z"}:
            opname = "nvvm.read.ptx.sreg.tid."
        else:
            raise NotImplementedError(op)
        res = self.scalar("i32")
        self.append(f"{res.name} = {opname + op[-1]} : {res}")
        return res

    def load(self, ptr: Pointer, offset: Scalar) -> Scalar:
        res = self.scalar("f32")
        info = f"{ptr}, {offset} -> {res}"
        self.append(f"{res.name} = oven.load {ptr.name}, {offset.name} : {info}")
        return res

    def store(self, ptr: Pointer, offset: Scalar, value: Scalar) -> None:
        info = f"{value}, {ptr}, {offset}"
        self.append(f"oven.store {value.name}, {ptr.name}, {offset.name} : {info}")


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

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.generic_visit(node)
        opname = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
        }[type(node.op)]
        left = self.get_value(node.target)
        right = self.get_value(node.value)
        result = self.writer.binary(opname, left, right)
        self.values[node.target.id] = result

    def visit_Return(self, node: ast.Return) -> None:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            values = [self.get_value(node.value.id)]
        elif isinstance(node.value, ast.Tuple):
            values = [self.get_value(elt) for elt in node.value.elts]
        else:
            raise NotImplementedError(type(node.value))
        self.writer.ret(values)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        arguments = []
        for arg in node.args.args:
            if isinstance(arg.annotation, ast.Name):
                assert arg.annotation.id in ("int", "float")
                dtype = {"int": "i32", "float": "f32"}[arg.annotation.id]
                value = self.writer.scalar(dtype)
            elif isinstance(arg.annotation, ast.Attribute):
                assert arg.annotation.value.id == "ol"
                assert arg.annotation.attr == "ptr"
                value = self.writer.pointer()
            else:
                value = self.writer.scalar("i32")

            self.values[arg.arg] = value
            arguments.append(f"{value.name}: {value}")
        self.writer.append(f"func.func @{node.name}({', '.join(arguments)}) {{")
        self.writer.indent += 1
        self.generic_visit(node)
        self.writer.indent -= 1
        self.writer.append("}")

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        assert isinstance(node.func, ast.Attribute)
        assert node.func.value.id == "ol"
        if len(node.args) == 0:
            self.values[node] = self.writer.get_op(node.func.attr)
        elif len((node.args)) == 1:
            arg = self.get_value(node.args[0])
            opname = {
                "sigmoid": "oven.sigmoid",
                "exp": "math.exp",
                "sqrt": "math.sqrt",
                "abs": "math.absf",
                "ceil": "math.ceil",
                "floor": "math.floor",
                "rsqrt": "oven.rsqrt",
            }.get(node.func.attr)
            if opname is None:
                raise NotImplementedError(node.func.attr)
            self.values[node] = self.writer.unary(opname, arg)
        elif len(node.args) == 2:
            if node.func.attr == "load":
                ptr = self.get_value(node.args[0])
                index = self.get_value(node.args[1])
                self.values[node] = self.writer.load(ptr, index)
            else:
                raise NotImplementedError(node.func.attr)
        elif len(node.args) == 3:
            if node.func.attr == "store":
                value = self.get_value(node.args[0])
                ptr = self.get_value(node.args[1])
                index = self.get_value(node.args[2])
                self.writer.store(ptr, index, value)
            elif node.func.attr == "vload":
                ptr = self.get_value(node.args[0])
                index = self.get_value(node.args[1])
                size = self.get_value(node.args[2])
                self.values[node] = self.writer.vload(ptr, index, size)
            else:
                raise NotImplementedError(node.func.attr)
