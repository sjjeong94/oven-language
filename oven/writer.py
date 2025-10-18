import ast
from typing import Any, List, Optional, Union, Tuple


def get_dtype(value: Any) -> str:
    dtype = {
        bool: "i1",
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
        if left.dtype == "index":
            left = self.to_i32(left)
        if right.dtype == "index":
            right = self.to_i32(right)
        assert left.dtype == right.dtype
        op = f"arith.{op}{left.dtype[0]}"
        if isinstance(left, Vector):
            res = self.vector(left.dtype, left.shape)
        elif isinstance(left, Scalar):
            res = self.scalar(left.dtype)
        else:
            raise NotImplementedError(type(left))
        self.append(f"{res.name} = {op} {left.name}, {right.name} : {left}")
        return res

    def compare(self, op: str, left: Value, right: Value) -> Value:
        if left.dtype == "index":
            left = self.to_i32(left)
        if right.dtype == "index":
            right = self.to_i32(right)
        assert left.dtype == right.dtype
        assert type(left) == type(right) == Scalar
        dtype = left.dtype[0]
        if dtype == "f":
            op = "o" + op
        elif dtype == "i" and op in {"lt", "le", "gt", "ge"}:
            op = "s" + op
        op = f"arith.cmp{dtype} {op},"
        res = self.scalar("i1")
        self.append(f"{res.name} = {op} {left.name}, {right.name} : {left}")
        return res

    def ret(self, values: List[Value]) -> None:
        line = f"return"
        if len(values) > 0:
            names = ", ".join(v.name for v in values)
            types = ", ".join(repr(v) for v in values)
            line += f" {names} : {types}"
        self.append(line)

    def get_op(self, op: str) -> Scalar:
        if op == "smem":
            res = self.pointer(space=3)
            self.append(f"{res.name} = oven.smem : {res}")
            return res
        elif op == "barrier":
            self.append("nvvm.barrier0")
            return

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
        info = f"({ptr}, {offset}) -> {res}"
        self.append(f"{res.name} = oven.load {ptr.name}, {offset.name} : {info}")
        return res

    def vload(self, ptr: Pointer, offset: Scalar) -> Vector:
        res = self.vector("f32", [4])
        info = f"({ptr}, {offset}) -> {res}"
        self.append(f"{res.name} = oven.vload {ptr.name}, {offset.name} : {info}")
        return res

    def store(self, value: Scalar, ptr: Pointer, offset: Scalar) -> None:
        assert isinstance(value, Scalar)
        info = f"({value}, {ptr}, {offset})"
        self.append(f"oven.store {value.name}, {ptr.name}, {offset.name} : {info}")

    def vstore(self, value: Vector, ptr: Pointer, offset: Scalar) -> None:
        assert isinstance(value, Vector)
        info = f"({value}, {ptr}, {offset})"
        self.append(f"oven.vstore {value.name}, {ptr.name}, {offset.name} : {info}")

    def to_index(self, value: Scalar) -> Scalar:
        assert value.dtype == "i32"
        res = self.scalar("index")
        self.append(f"{res.name} = arith.index_cast {value.name} : {value} to {res}")
        return res

    def to_i32(self, value: Scalar) -> Scalar:
        assert value.dtype == "index"
        res = self.scalar("i32")
        self.append(f"{res.name} = arith.index_cast {value.name} : {value} to {res}")
        return res

    def scf_for(
        self,
        index: Scalar,
        start: Scalar,
        end: Scalar,
        step: Scalar,
        iter_args: List[Tuple[Value, Value]],
    ) -> List[Value]:
        args = [f"{new_arg.name} = {arg.name}" for arg, new_arg in iter_args]
        args = ", ".join(args)
        info = ", ".join([repr(arg) for arg, _ in iter_args])
        results = [self.scalar(arg.dtype) for arg, _ in iter_args]

        res_names = ", ".join([res.name for res in results])
        op = f"scf.for {index.name} = {start.name} to {end.name} step {step.name}"
        line = f"{res_names} = {op} iter_args({args}) -> ({info}) {{"
        self.append(line)
        return results

    def scf_yield(self, values: List[Value]) -> None:
        names = ", ".join(v.name for v in values)
        types = ", ".join(repr(v) for v in values)
        self.append(f"scf.yield {names} : {types}")


class Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.writer = Writer()
        self.values = dict()

    def __repr__(self):
        return repr(self.writer)

    def get_value(self, node: Union[ast.AST, int]) -> Value:
        if isinstance(node, int):
            return self.values[node]
        elif isinstance(node, ast.Name):
            return self.values[node.id]
        return self.values[node]

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        if isinstance(node.targets[0], ast.Tuple):
            assert isinstance(node.value, ast.Tuple)
            for target, value in zip(node.targets[0].elts, node.value.elts):
                assert isinstance(target, ast.Name)
                self.values[target.id] = self.values[value]
        elif isinstance(node.targets[0], ast.Name):
            self.values[node.targets[0].id] = self.values[node.value]
        else:
            raise NotImplementedError(type(node.targets[0]))

    def visit_Constant(self, node: ast.Constant) -> None:
        self.generic_visit(node)
        self.values[node] = self.writer.constant(node.value)

    def visit_Compare(self, node: ast.Compare) -> None:
        assert len(node.ops) == 1
        self.generic_visit(node)
        left = self.get_value(node.left)
        right = self.get_value(node.comparators[0])
        opname = {
            ast.Eq: "eq",
            ast.NotEq: "ne",
            ast.Lt: "lt",
            ast.LtE: "le",
            ast.Gt: "gt",
            ast.GtE: "ge",
        }[type(node.ops[0])]
        result = self.writer.compare(opname, left, right)
        self.values[node] = result

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
        elif isinstance(node.value, ast.Constant):
            values = [self.get_value(node.value)]
        elif node.value is None:
            values = []
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
        if self.writer.lines[-1].split()[0] != "return":
            self.writer.ret([])
        self.writer.indent -= 1
        self.writer.append("}")

    def visit_range(self, node: ast.Call) -> None:
        if len(node.args) == 3:
            start = self.get_value(node.args[0])
            end = self.get_value(node.args[1])
            step = self.get_value(node.args[2])
        elif len(node.args) == 2:
            start = self.get_value(node.args[0])
            end = self.get_value(node.args[1])
            step = self.writer.constant(1)
        elif len(node.args) == 1:
            start = self.writer.constant(0)
            end = self.get_value(node.args[0])
            step = self.writer.constant(1)
        else:
            raise NotImplementedError(len(node.args))
        self.values[node] = (start, end, step)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                self.visit_range(node)
                return
            else:
                raise NotImplementedError(node.func.id)

        assert isinstance(node.func, ast.Attribute), node.func
        assert node.func.value.id == "ol"

        if len(node.args) == 0:
            self.values[node] = self.writer.get_op(node.func.attr)
        elif node.func.attr == "range":
            self.visit_range(node)
        elif len((node.args)) == 1:
            arg = self.get_value(node.args[0])
            opname = {
                "sigmoid": "oven.sigmoid",
                "exp2": "math.exp2",
                "exp": "math.exp",
                "sqrt": "math.sqrt",
                "abs": "math.absf",
                "ceil": "math.ceil",
                "floor": "math.floor",
                "rsqrt": "math.rsqrt",
            }.get(node.func.attr)
            if opname is None:
                raise NotImplementedError(node.func.attr)
            self.values[node] = self.writer.unary(opname, arg)
        elif len(node.args) == 2:
            if node.func.attr == "load":
                ptr = self.get_value(node.args[0])
                offset = self.get_value(node.args[1])
                self.values[node] = self.writer.load(ptr, offset)
            elif node.func.attr == "vload":
                ptr = self.get_value(node.args[0])
                offset = self.get_value(node.args[1])
                self.values[node] = self.writer.vload(ptr, offset)
            else:
                raise NotImplementedError(node.func.attr)
        elif len(node.args) == 3:
            if node.func.attr == "store":
                value = self.get_value(node.args[0])
                ptr = self.get_value(node.args[1])
                offset = self.get_value(node.args[2])
                self.writer.store(value, ptr, offset)
            elif node.func.attr == "vstore":
                value = self.get_value(node.args[0])
                ptr = self.get_value(node.args[1])
                offset = self.get_value(node.args[2])
                self.writer.vstore(value, ptr, offset)
            else:
                raise NotImplementedError(node.func.attr)

    def visit_For(self, node: ast.For) -> None:
        index = self.writer.scalar("index")
        self.values[node.target.id] = index
        self.visit(node.iter)
        start, end, step = self.get_value(node.iter)
        start = self.writer.to_index(start)
        end = self.writer.to_index(end)
        step = self.writer.to_index(step)

        ids = []
        for b in node.body:
            if isinstance(b, ast.AugAssign):
                if b.target.id not in ids:
                    ids.append(b.target.id)
            elif isinstance(b, ast.Assign) and isinstance(b.value, ast.BinOp):
                if (
                    isinstance(b.value.left, ast.Name)
                    and b.value.left.id == b.targets[0].id
                    and b.value.left.id not in ids
                ):
                    ids.append(b.value.left.id)
                if (
                    isinstance(b.value.right, ast.Name)
                    and b.value.right.id == b.targets[0].id
                    and b.value.right.id not in ids
                ):
                    ids.append(b.value.right.id)

        iter_args = []
        for i in ids:
            arg = self.get_value(i)
            new_arg = self.writer.scalar(arg.dtype)
            self.values[i] = new_arg
            iter_args.append((arg, new_arg))

        results = self.writer.scf_for(index, start, end, step, iter_args)
        self.writer.indent += 1
        for b in node.body:
            self.visit(b)
        r = [self.get_value(i) for i in ids]
        self.writer.scf_yield(r)
        for i, res in zip(ids, results):
            self.values[i] = res
        self.writer.indent -= 1
        self.writer.append("}")

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        cond = self.get_value(node.test)
        self.writer.append(f"scf.if {cond.name} {{")
        self.writer.indent += 1
        for b in node.body:
            self.visit(b)
        self.writer.indent -= 1
        if node.orelse:
            self.writer.append("} else {")
            self.writer.indent += 1
            for b in node.orelse:
                self.visit(b)
            self.writer.indent -= 1
        self.writer.append("}")
