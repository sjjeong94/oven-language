"""
Simplified test suite for oven language compiler operations.

This module contains working unit and integration tests for the oven language compiler.
"""

import pytest
import ast
import os
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
class TestCompilerBasics:
    """Test basic compiler functionality."""

    def test_compiler_initialization(self, compiler):
        """Test that compiler initializes correctly."""
        assert compiler is not None
        assert hasattr(compiler, "compile_source")

    def test_writer_visitor_initialization(self, writer_visitor):
        """Test that writer visitor initializes correctly."""
        assert writer_visitor is not None
        assert hasattr(writer_visitor, "writer")
        assert hasattr(writer_visitor.writer, "lines")


@pytest.mark.unit
class TestUnaryOperations:
    """Test unary mathematical operations compilation."""

    @pytest.mark.parametrize(
        "func_name,expected_pattern_options",
        [
            ("sigmoid", ["oven.sigmoid", "math.sigmoid"]),
            ("exp", ["math.exp", "oven.exp"]),
            ("sqrt", ["math.sqrt", "oven.sqrt"]),
            ("abs", ["math.absf", "oven.abs"]),
            ("ceil", ["math.ceil", "oven.ceil"]),
            ("floor", ["math.floor", "oven.floor"]),
            ("rsqrt", ["math.rsqrt", "oven.rsqrt"]),
        ],
    )
    def test_unary_operations_compilation(
        self, compiler, func_name, expected_pattern_options
    ):
        """Test compilation of various unary operations."""
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

        # Check for function declaration
        assert f"func.func @{func_name}" in mlir_code

        # Check for operation pattern (flexible matching)
        pattern_found = any(
            pattern in mlir_code for pattern in expected_pattern_options
        )
        assert pattern_found, f"No expected pattern found for {func_name} in MLIR code"


@pytest.mark.unit
class TestBinaryOperations:
    """Test binary mathematical operations compilation."""

    @pytest.mark.parametrize(
        "operation,op_symbol,expected_pattern_options",
        [
            ("add", "+", ["arith.addf", "arith.addi"]),
            ("mul", "*", ["arith.mulf", "arith.muli"]),
            ("sub", "-", ["arith.subf", "arith.subi"]),
            ("div", "/", ["arith.divf"]),  # Division is typically float
        ],
    )
    def test_binary_operations_compilation(
        self, compiler, operation, op_symbol, expected_pattern_options
    ):
        """Test compilation of various binary operations."""
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

        # Check for function declaration
        assert f"func.func @{operation}" in mlir_code

        # Check for operation pattern (flexible matching)
        pattern_found = any(
            pattern in mlir_code for pattern in expected_pattern_options
        )
        assert pattern_found, f"No expected pattern found for {operation} in MLIR code"


@pytest.mark.unit
class TestMatrixOperations:
    """Test matrix multiplication operations compilation."""

    def test_matmul_compilation(self, compiler):
        """Test compilation of matrix multiplication."""
        source = """
import oven.language as ol

def matmul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, m: int, n: int, k: int):
    block_size = ol.get_bdim_x()
    cCol = ol.get_bid_x()
    cRow = ol.get_bid_y()
    tCol = ol.get_tid_x()
    tRow = ol.get_tid_y()

    col = cCol * block_size + tCol
    row = cRow * block_size + tRow

    acc = 0.0
    for i in range(0, k, 1):
        a_offset = row * k + i
        b_offset = i * n + col

        a_val = ol.load(a_ptr, a_offset)
        b_val = ol.load(b_ptr, b_offset)
        acc = acc + a_val * b_val

    c_offset = row * n + col
    ol.store(acc, c_ptr, c_offset)
"""
        mlir_code = compiler.compile_source(source)

        # Check for expected MLIR patterns
        assert "func.func @matmul" in mlir_code
        assert "scf.for" in mlir_code

        # Check for arithmetic operations
        arithmetic_ops = ["arith.addf", "arith.mulf", "arith.addi", "arith.muli"]
        arithmetic_found = any(op in mlir_code for op in arithmetic_ops)
        assert arithmetic_found, "No arithmetic operations found"


@pytest.mark.unit
class TestWriterOperations:
    """Test operations using the writer visitor directly."""

    def test_simple_arithmetic_with_writer(self, writer_visitor):
        """Test simple arithmetic operations with writer visitor."""
        test_cases = [
            ("a = 5; b = 3; c = a + b", "arith.addi"),
            ("a = 5; b = 3; c = a - b", "arith.subi"),
            ("a = 5; b = 3; c = a * b", "arith.muli"),
        ]

        for source_code, expected_op in test_cases:
            # Reset visitor for each test
            visitor = Visitor()
            tree = ast.parse(source_code, filename="<test>")
            visitor.visit(tree)

            operations_str = "\n".join(visitor.writer.lines)
            assert (
                expected_op in operations_str
            ), f"Expected {expected_op} not found in {operations_str}"

    def test_constants_with_writer(self, writer_visitor):
        """Test constant generation with writer visitor."""
        source = """
a = 42
b = 3.14
c = True
"""
        tree = ast.parse(source, filename="<test>")
        writer_visitor.visit(tree)

        operations_str = "\n".join(writer_visitor.writer.lines)

        # Check for different constant types
        assert "arith.constant 42" in operations_str
        assert "arith.constant 3.14" in operations_str
        assert "arith.constant True" in operations_str


@pytest.mark.integration
class TestFileCompilation:
    """Integration tests for compiling example files."""

    def test_compile_unary_op_file(self, compiler):
        """Test compiling the unary_op.py example file."""
        file_path = "/Users/jsj/projects/oven-language/tests/unary_op.py"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                source_code = f.read()

            mlir_code = compiler.compile_source(source_code)

            # Basic sanity checks
            assert mlir_code is not None
            assert len(mlir_code) > 0

            # Check that expected functions are present
            expected_functions = [
                "sigmoid",
                "exp",
                "sqrt",
                "abs",
                "ceil",
                "floor",
                "rsqrt",
            ]
            for func_name in expected_functions:
                assert (
                    f"func.func @{func_name}" in mlir_code
                ), f"Function {func_name} not found"

    def test_compile_binary_op_file(self, compiler):
        """Test compiling the binary_op.py example file."""
        file_path = "/Users/jsj/projects/oven-language/tests/binary_op.py"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                source_code = f.read()

            mlir_code = compiler.compile_source(source_code)

            # Basic sanity checks
            assert mlir_code is not None
            assert len(mlir_code) > 0

            # Check that expected functions are present
            expected_functions = ["add", "mul", "sub", "div"]
            for func_name in expected_functions:
                assert (
                    f"func.func @{func_name}" in mlir_code
                ), f"Function {func_name} not found"

    def test_compile_matmul_op_file(self, compiler):
        """Test compiling the matmul_op.py example file."""
        file_path = "/Users/jsj/projects/oven-language/tests/matmul_op.py"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                source_code = f.read()

            mlir_code = compiler.compile_source(source_code)

            # Basic sanity checks
            assert mlir_code is not None
            assert len(mlir_code) > 0

            # Check that the matmul function is present
            assert "func.func @matmul" in mlir_code
            assert "scf.for" in mlir_code


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in compilation."""

    def test_syntax_error_handling(self, compiler):
        """Test handling of Python syntax errors."""
        invalid_source = """
def broken_function(
    # Missing closing parenthesis and colon
"""
        with pytest.raises(SyntaxError):
            compiler.compile_source(invalid_source)

    def test_empty_source_handling(self, compiler):
        """Test handling of empty source code."""
        empty_source = ""
        mlir_code = compiler.compile_source(empty_source)

        # Should handle gracefully (might return empty string or minimal MLIR)
        assert isinstance(mlir_code, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
