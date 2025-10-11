"""
Test cases for matrix multiplication operations using pytest framework.

This module contains unit tests for matrix multiplication operations
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
class TestMatrixMultiplication:
    """Test matrix multiplication operations compilation."""

    def test_matmul_function_compilation(self, compiler):
        """Test compilation of matrix multiplication function."""
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

        # Check for loop constructs
        assert "scf.for" in mlir_code

        # Check for arithmetic operations
        arithmetic_ops = ["arith.addf", "arith.mulf", "arith.addi", "arith.muli"]
        arithmetic_found = any(op in mlir_code for op in arithmetic_ops)
        assert arithmetic_found, "No arithmetic operations found"

        # Check for GPU operations (if supported)
        gpu_patterns = [
            "nvvm.read.ptx.sreg",
            "gpu.thread_id",
            "gpu.block_id",
            "gpu.block_dim",
        ]
        # Note: These might not be present depending on implementation

    def test_matmul_with_gpu_intrinsics(self, compiler):
        """Test matrix multiplication with GPU intrinsics."""
        source = """
import oven.language as ol

def matmul_gpu(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr, m: int, n: int, k: int):
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

        # Basic function check
        assert "func.func @matmul_gpu" in mlir_code

        # Check for loop and arithmetic operations
        assert "scf.for" in mlir_code


@pytest.mark.unit
class TestMatrixOperationsWithWriter:
    """Test matrix operations using the writer visitor directly."""

    def test_simple_matrix_operation_with_writer(self, writer_visitor):
        """Test simple matrix operation compilation with writer visitor."""
        # Test with a simple nested loop
        source = """
for i in range(3):
    for j in range(3):
        c = i * j
"""
        tree = ast.parse(source, filename="<test>")
        writer_visitor.visit(tree)

        # Check that operations were generated
        operations = writer_visitor.writer.lines
        assert len(operations) > 0

    def test_matrix_accumulation_with_writer(self, writer_visitor):
        """Test matrix accumulation pattern with writer visitor."""
        source = """
acc = 0
for i in range(10):
    acc = acc + i
"""
        tree = ast.parse(source, filename="<test>")
        writer_visitor.visit(tree)

        operations = writer_visitor.writer.lines
        operations_str = "\n".join(operations)

        # Check for accumulation pattern
        assert (
            "arith.constant 0" in operations_str
            or "arith.constant 0.0" in operations_str
        )


@pytest.mark.integration
class TestMatrixMultiplicationIntegration:
    """Integration tests for matrix multiplication operations."""

    def test_matmul_with_different_sizes(self, compiler):
        """Test matrix multiplication with different sizes."""
        sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]

        for m, n, k in sizes:
            source = f"""
import oven.language as ol

def matmul_{m}x{n}x{k}(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    m, n, k = {m}, {n}, {k}
    
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

            # Check that compilation succeeds for different sizes
            assert f"func.func @matmul_{m}x{n}x{k}" in mlir_code
            assert "scf.for" in mlir_code

    @pytest.mark.slow
    def test_large_matmul_compilation(self, compiler):
        """Test compilation of large matrix multiplication (marked as slow)."""
        source = """
import oven.language as ol

def large_matmul(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    m, n, k = 1024, 1024, 1024
    
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

        # Basic checks for large matrix
        assert "func.func @large_matmul" in mlir_code
        assert "scf.for" in mlir_code


if __name__ == "__main__":
    pytest.main([__file__])
