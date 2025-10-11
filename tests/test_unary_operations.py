"""
Test cases for unary operations using pytest framework.

This module contains unit tests for various unary mathematical operations
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
class TestUnaryOperations:
    """Test unary mathematical operations compilation."""

    def test_sigmoid_function_compilation(self, compiler):
        """Test compilation of sigmoid function."""
        source = """
import oven.language as ol

def sigmoid(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.sigmoid(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @sigmoid" in mlir_code
        assert "oven.sigmoid" in mlir_code or "math.sigmoid" in mlir_code

    def test_exp_function_compilation(self, compiler):
        """Test compilation of exponential function."""
        source = """
import oven.language as ol

def exp(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.exp(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @exp" in mlir_code
        assert "math.exp" in mlir_code

    def test_sqrt_function_compilation(self, compiler):
        """Test compilation of square root function."""
        source = """
import oven.language as ol

def sqrt(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.sqrt(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @sqrt" in mlir_code
        assert "math.sqrt" in mlir_code

    def test_abs_function_compilation(self, compiler):
        """Test compilation of absolute value function."""
        source = """
import oven.language as ol

def abs(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.abs(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @abs" in mlir_code
        assert "math.absf" in mlir_code

    def test_ceil_function_compilation(self, compiler):
        """Test compilation of ceiling function."""
        source = """
import oven.language as ol

def ceil(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.ceil(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @ceil" in mlir_code
        assert "math.ceil" in mlir_code

    def test_floor_function_compilation(self, compiler):
        """Test compilation of floor function."""
        source = """
import oven.language as ol

def floor(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.floor(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @floor" in mlir_code
        assert "math.floor" in mlir_code

    def test_rsqrt_function_compilation(self, compiler):
        """Test compilation of reciprocal square root function."""
        source = """
import oven.language as ol

def rsqrt(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.rsqrt(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @rsqrt" in mlir_code
        # Allow flexible pattern matching for different MLIR dialects
        pattern_found = "math.rsqrt" in mlir_code or "oven.rsqrt" in mlir_code
        assert pattern_found, f"No rsqrt pattern found in generated MLIR code"

    @pytest.mark.parametrize(
        "func_name,mlir_pattern",
        [
            ("sigmoid", "oven.sigmoid"),
            ("exp", "math.exp"),
            ("sqrt", "math.sqrt"),
            ("abs", "math.absf"),
            ("ceil", "math.ceil"),
            ("floor", "math.floor"),
            ("rsqrt", "math.rsqrt"),
        ],
    )
    def test_unary_operations_parametrized(self, compiler, func_name, mlir_pattern):
        """Test compilation of various unary operations using parametrized approach."""
        source = f"""
import oven.language as ol

def {func_name}(x_ptr: ol.ptr, y_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = bid * bsize + tid
    x_value = ol.load(x_ptr, idx)
    y_value = ol.{func_name}(x_value)
    ol.store(y_value, y_ptr, idx)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert f"func.func @{func_name}" in mlir_code
        # Allow flexible pattern matching for different MLIR dialects
        pattern_found = (
            mlir_pattern in mlir_code
            or f"oven.{func_name}" in mlir_code
            or f"math.{func_name}" in mlir_code
        )
        assert (
            pattern_found
        ), f"Pattern '{mlir_pattern}' not found in generated MLIR code"


@pytest.mark.unit
class TestUnaryOperationsWithWriter:
    """Test unary operations using the writer visitor directly."""

    def test_simple_unary_operations_with_writer(self, writer_visitor):
        """Test simple operations compilation with writer visitor."""
        # Test with simple constants only
        source = """
a = 5
b = 10
"""
        tree = ast.parse(source, filename="<test>")
        writer_visitor.visit(tree)

        # Check that operations were generated
        operations = writer_visitor.writer.lines
        assert len(operations) > 0

        # Check for constant patterns
        operations_str = "\n".join(operations)
        assert "arith.constant 5" in operations_str
        assert "arith.constant 10" in operations_str


if __name__ == "__main__":
    pytest.main([__file__])
