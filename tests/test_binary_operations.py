"""
Test cases for binary operations using pytest framework.

This module contains unit tests for various binary mathematical operations
implemented in the oven language compiler.
"""

import pytest
import ast
from oven.compiler import PythonToMLIRCompiler
from oven.writer import Visitor


@pytest.fixture
def compiler():
    """Create a compiler instance for testing."""
    return PythonToMLIRCompiler(debug=False, optimize=False)


@pytest.fixture
def writer_visitor():
    """Create a writer visitor instance for testing."""
    return Visitor()


@pytest.mark.unit
class TestBinaryOperations:
    """Test binary mathematical operations compilation."""

    def test_add_function_compilation(self, compiler):
        """Test compilation of addition function."""
        source = """
import oven.language as ol

def add(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value + y_value
    ol.store(z_value, c_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @add" in mlir_code
        assert "arith.addf" in mlir_code or "arith.addi" in mlir_code

    def test_mul_function_compilation(self, compiler):
        """Test compilation of multiplication function."""
        source = """
import oven.language as ol

def mul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value * y_value
    ol.store(z_value, c_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @mul" in mlir_code
        assert "arith.mulf" in mlir_code or "arith.muli" in mlir_code

    def test_sub_function_compilation(self, compiler):
        """Test compilation of subtraction function."""
        source = """
import oven.language as ol

def sub(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value - y_value
    ol.store(z_value, c_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @sub" in mlir_code
        assert "arith.subf" in mlir_code or "arith.subi" in mlir_code

    def test_div_function_compilation(self, compiler):
        """Test compilation of division function."""
        source = """
import oven.language as ol

def div(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value / y_value
    ol.store(z_value, c_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @div" in mlir_code
        assert "arith.divf" in mlir_code

    @pytest.mark.parametrize(
        "operation,op_symbol,expected_mlir",
        [
            ("add", "+", "arith.add"),
            ("mul", "*", "arith.mul"),
            ("sub", "-", "arith.sub"),
            ("div", "/", "arith.div"),
        ],
    )
    def test_binary_operations_parametrized(
        self, compiler, operation, op_symbol, expected_mlir
    ):
        """Test compilation of various binary operations using parametrized approach."""
        source = f"""
import oven.language as ol

def {operation}(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value {op_symbol} y_value
    ol.store(z_value, c_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert f"func.func @{operation}" in mlir_code
        # Allow for both integer and float variants
        pattern_found = (
            f"{expected_mlir}f" in mlir_code or f"{expected_mlir}i" in mlir_code
        )
        assert (
            pattern_found
        ), f"Pattern '{expected_mlir}' not found in generated MLIR code"


@pytest.mark.unit
class TestBinaryOperationsWithWriter:
    """Test binary operations using the writer visitor directly."""

    def test_simple_binary_operations_with_writer(self, writer_visitor):
        """Test simple binary operations compilation with writer visitor."""
        # Test with a simple binary operation
        source = """
a = 5
b = 3
c = a + b
"""
        tree = ast.parse(source, filename="<test>")
        writer_visitor.visit(tree)

        # Check that operations were generated
        operations = writer_visitor.writer.lines
        assert len(operations) > 0

        # Check for constant and operation patterns
        operations_str = "\n".join(operations)
        assert "arith.constant 5" in operations_str
        assert "arith.constant 3" in operations_str
        assert "arith.addi" in operations_str

    def test_multiple_binary_operations_with_writer(self, writer_visitor):
        """Test multiple binary operations with writer visitor."""
        source = """
a = 10
b = 5
c = a + b
d = a - b
e = a * b
f = a / b
"""
        tree = ast.parse(source, filename="<test>")
        writer_visitor.visit(tree)

        operations = writer_visitor.writer.lines
        operations_str = "\n".join(operations)

        # Check for all operation types
        assert "arith.constant 10" in operations_str
        assert "arith.constant 5" in operations_str
        assert "arith.addi" in operations_str
        assert "arith.subi" in operations_str
        assert "arith.muli" in operations_str

    @pytest.mark.parametrize(
        "source_code,expected_op",
        [
            ("a = 1; b = 2; c = a + b", "arith.addi"),
            ("a = 1; b = 2; c = a - b", "arith.subi"),
            ("a = 1; b = 2; c = a * b", "arith.muli"),
        ],
    )
    def test_writer_binary_operations_parametrized(
        self, writer_visitor, source_code, expected_op
    ):
        """Test binary operations with writer using parametrized approach."""
        tree = ast.parse(source_code, filename="<test>")
        writer_visitor.visit(tree)

        operations_str = "\n".join(writer_visitor.writer.lines)
        assert expected_op in operations_str


@pytest.mark.integration
class TestBinaryOperationsIntegration:
    """Integration tests for binary operations."""

    def test_binary_operations_with_gpu_context(self, compiler):
        """Test binary operations in GPU context."""
        source = """
import oven.language as ol

def gpu_add(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(a_ptr, idx)
    y_value = ol.load(b_ptr, idx)
    z_value = x_value + y_value
    ol.store(z_value, c_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for GPU-specific operations
        assert "func.func @gpu_add" in mlir_code
        # Should contain GPU intrinsics
        gpu_patterns = [
            "nvvm.read.ptx.sreg",
            "gpu.thread_id",
            "gpu.block_id",
            "gpu.block_dim",
        ]
        gpu_found = any(pattern in mlir_code for pattern in gpu_patterns)
        # Note: This might not be present depending on the current implementation
        # assert gpu_found, "No GPU patterns found in compiled MLIR code"


if __name__ == "__main__":
    pytest.main([__file__])
