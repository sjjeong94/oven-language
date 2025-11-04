# Oven Language

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/oven-language.svg)](https://badge.fury.io/py/oven-language)
[![Tests](https://github.com/sjjeong94/oven-language/workflows/Tests/badge.svg)](https://github.com/sjjeong94/oven-language/actions)
[![Build](https://github.com/sjjeong94/oven-language/workflows/Build/badge.svg)](https://github.com/sjjeong94/oven-language/actions)

A Python-to-MLIR compiler for GPU kernel development.

## Installation

```bash
pip install oven-language
```

## Quick Start

```python
import oven.language as ol

def vector_add(a_ptr: ol.ptr, b_ptr: ol.ptr, c_ptr: ol.ptr):
    bsize = ol.get_bdim_x()
    bid = ol.get_bid_x()
    tid = ol.get_tid_x()
    idx = (bid * bsize + tid) * 4
    x_value = ol.vload(a_ptr, idx)
    y_value = ol.vload(b_ptr, idx)
    z_value = x_value + y_value
    ol.vstore(z_value, c_ptr, idx)
```

Compile to MLIR:
```bash
oven kernel.py
```

## Key Functions

- **Thread/Block IDs**: `ol.get_tid_x()`, `ol.get_bid_x()`
- **Memory**: `ol.load()`, `ol.store()`, `ol.vload()`, `ol.vstore()`
- **Math**: `ol.exp()`, `ol.log()`, `ol.sin()`, `ol.sigmoid()`
- **GPU**: `ol.barrier()`, `ol.smem()`

## Links

- [GitHub](https://github.com/sjjeong94/oven-language)
- [PyPI](https://pypi.org/project/oven-language/)
