#!/usr/bin/env python3
"""
Detailed investigation of Python allocation patterns.

This script explores specific scenarios in depth, including:
- Integer freelist behavior
- String interning effects
- Container resizing patterns
- The difference between creating and reusing objects
"""

import tracemalloc
import sys
import gc
from typing import List, Tuple


def measure_allocations(operation, iterations: int = 100000) -> Tuple[int, int]:
    """
    Measure total allocations for an operation.

    Returns:
        Tuple of (total_bytes_allocated, total_allocation_count)
    """
    gc.collect()
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    operation(iterations)

    snapshot2 = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_allocated = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    total_blocks = sum(stat.count_diff for stat in stats if stat.count_diff > 0)

    return total_allocated, total_blocks


def test_integer_arithmetic_patterns():
    """Explore integer allocation patterns similar to the blog post."""
    print("\n" + "=" * 80)
    print("INTEGER ARITHMETIC PATTERNS")
    print("=" * 80)

    # Test 1: Simple addition without storing result
    def simple_addition(n):
        x = 0
        for i in range(n):
            x + 1  # Result is discarded

    bytes_allocated, alloc_count = measure_allocations(simple_addition)
    print(f"\n1. Simple addition (result discarded):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 2: Addition with result stored
    def addition_with_storage(n):
        x = 0
        for i in range(n):
            x = x + 1

    bytes_allocated, alloc_count = measure_allocations(addition_with_storage)
    print(f"\n2. Addition with result stored:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 3: Addition with print (forces object creation)
    def addition_with_print(n):
        x = 0
        for i in range(n):
            result = x + 1
            str(result)  # Simulate conversion without actual print

    bytes_allocated, alloc_count = measure_allocations(addition_with_print)
    print(f"\n3. Addition with string conversion:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 4: Integer creation in small range
    def small_int_creation(n):
        for i in range(n):
            x = i % 256  # Should use pre-allocated integers

    bytes_allocated, alloc_count = measure_allocations(small_int_creation)
    print(f"\n4. Small integer creation (0-255 range):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 5: Integer creation outside small range
    def large_int_creation(n):
        for i in range(n):
            x = i + 1000  # Outside pre-allocated range

    bytes_allocated, alloc_count = measure_allocations(large_int_creation)
    print(f"\n5. Large integer creation (>256):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 6: Very large integer operations
    def very_large_int_ops(n):
        big = 10**100
        for i in range(n):
            big + 1

    bytes_allocated, alloc_count = measure_allocations(very_large_int_ops)
    print(f"\n6. Very large integer operations (10^100):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")


def test_string_allocation_patterns():
    """Explore string allocation and interning."""
    print("\n" + "=" * 80)
    print("STRING ALLOCATION PATTERNS")
    print("=" * 80)

    # Test 1: String concatenation (always allocates)
    def string_concat(n):
        s = "hello"
        for i in range(n):
            result = s + "world"

    bytes_allocated, alloc_count = measure_allocations(string_concat)
    print(f"\n1. String concatenation:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 2: String formatting with f-strings
    def fstring_format(n):
        for i in range(n):
            s = f"value: {i}"

    bytes_allocated, alloc_count = measure_allocations(fstring_format)
    print(f"\n2. F-string formatting:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 3: String interning
    def string_intern(n):
        for i in range(n):
            s = sys.intern(f"str_{i % 100}")  # Reuse 100 strings

    bytes_allocated, alloc_count = measure_allocations(string_intern)
    print(f"\n3. String interning (100 unique strings):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 4: String slicing
    def string_slice(n):
        s = "a" * 1000
        for i in range(n):
            result = s[10:20]

    bytes_allocated, alloc_count = measure_allocations(string_slice)
    print(f"\n4. String slicing:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 5: String repetition
    def string_repeat(n):
        for i in range(n):
            s = "x" * 10

    bytes_allocated, alloc_count = measure_allocations(string_repeat)
    print(f"\n5. String repetition:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")


def test_container_growth_patterns():
    """Explore how containers allocate as they grow."""
    print("\n" + "=" * 80)
    print("CONTAINER GROWTH PATTERNS")
    print("=" * 80)

    # Test 1: List growth via append
    def list_append_growth(n):
        lst = []
        for i in range(n):
            lst.append(i)

    bytes_allocated, alloc_count = measure_allocations(list_append_growth)
    print(f"\n1. List growth via append (0 to {100000:,} items):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 2: List with pre-allocation
    def list_preallocated(n):
        lst = [None] * n
        for i in range(n):
            lst[i] = i

    bytes_allocated, alloc_count = measure_allocations(list_preallocated)
    print(f"\n2. List with pre-allocation:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 3: List comprehension (size known upfront)
    def list_comprehension(n):
        for _ in range(1000):  # Reduced iterations for comprehension
            lst = [i for i in range(100)]

    bytes_allocated, alloc_count = measure_allocations(list_comprehension, iterations=1000)
    print(f"\n3. List comprehension (100 items, 1000 iterations):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/1000:.6f}")

    # Test 4: Dict growth via insertion
    def dict_insert_growth(n):
        d = {}
        for i in range(n):
            d[i] = i

    bytes_allocated, alloc_count = measure_allocations(dict_insert_growth)
    print(f"\n4. Dict growth via insertion (0 to {100000:,} items):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 5: Set growth
    def set_growth(n):
        s = set()
        for i in range(n):
            s.add(i)

    bytes_allocated, alloc_count = measure_allocations(set_growth)
    print(f"\n5. Set growth via add (0 to {100000:,} items):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")


def test_object_creation_patterns():
    """Test object creation and memory patterns."""
    print("\n" + "=" * 80)
    print("OBJECT CREATION PATTERNS")
    print("=" * 80)

    # Test 1: Simple class instance
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def create_instances(n):
        for i in range(n):
            p = Point(i, i+1)

    bytes_allocated, alloc_count = measure_allocations(create_instances, iterations=10000)
    print(f"\n1. Simple class instances (10,000 iterations):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/10000:.6f}")

    # Test 2: Class with __slots__
    class PointSlots:
        __slots__ = ['x', 'y']
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def create_slotted_instances(n):
        for i in range(n):
            p = PointSlots(i, i+1)

    bytes_allocated, alloc_count = measure_allocations(create_slotted_instances, iterations=10000)
    print(f"\n2. Class instances with __slots__ (10,000 iterations):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/10000:.6f}")

    # Test 3: Tuple creation (immutable)
    def create_tuples(n):
        for i in range(n):
            t = (i, i+1, i+2)

    bytes_allocated, alloc_count = measure_allocations(create_tuples)
    print(f"\n3. Tuple creation (100,000 iterations):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 4: Named tuple creation
    from collections import namedtuple
    PointNT = namedtuple('PointNT', ['x', 'y'])

    def create_namedtuples(n):
        for i in range(n):
            p = PointNT(i, i+1)

    bytes_allocated, alloc_count = measure_allocations(create_namedtuples, iterations=10000)
    print(f"\n4. Named tuple creation (10,000 iterations):")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/10000:.6f}")


def test_function_call_overhead():
    """Test allocation overhead of function calls."""
    print("\n" + "=" * 80)
    print("FUNCTION CALL OVERHEAD")
    print("=" * 80)

    # Test 1: Simple function
    def simple_func(x):
        return x + 1

    def call_simple(n):
        for i in range(n):
            result = simple_func(i)

    bytes_allocated, alloc_count = measure_allocations(call_simple)
    print(f"\n1. Simple function calls:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 2: Function with *args
    def varargs_func(*args):
        return sum(args)

    def call_varargs(n):
        for i in range(n):
            result = varargs_func(i, i+1, i+2)

    bytes_allocated, alloc_count = measure_allocations(call_varargs)
    print(f"\n2. Function with *args:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 3: Function with **kwargs
    def kwargs_func(**kwargs):
        return sum(kwargs.values())

    def call_kwargs(n):
        for i in range(n):
            result = kwargs_func(a=i, b=i+1, c=i+2)

    bytes_allocated, alloc_count = measure_allocations(call_kwargs)
    print(f"\n3. Function with **kwargs:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")

    # Test 4: Lambda
    lam = lambda x: x + 1

    def call_lambda(n):
        for i in range(n):
            result = lam(i)

    bytes_allocated, alloc_count = measure_allocations(call_lambda)
    print(f"\n4. Lambda calls:")
    print(f"   Total allocations: {alloc_count:,}")
    print(f"   Bytes allocated: {bytes_allocated:,}")
    print(f"   Allocations per iteration: {alloc_count/100000:.6f}")


def main():
    print("=" * 80)
    print("DETAILED PYTHON MEMORY ALLOCATION STUDY")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print()

    test_integer_arithmetic_patterns()
    test_string_allocation_patterns()
    test_container_growth_patterns()
    test_object_creation_patterns()
    test_function_call_overhead()

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
1. INTEGER OPERATIONS:
   - Small integers (-5 to 256) are pre-allocated and don't trigger allocations
   - Integer arithmetic often reuses objects from the freelist
   - Very few allocations per iteration for simple arithmetic

2. STRING OPERATIONS:
   - String concatenation and formatting create new objects (immutable)
   - String interning can reduce allocations for repeated strings
   - Each string operation typically allocates

3. CONTAINER GROWTH:
   - Lists and dicts allocate in chunks as they grow (not every append)
   - Pre-allocated containers are more efficient
   - Growth allocations are amortized O(1)

4. OBJECT CREATION:
   - Each object creation allocates memory
   - __slots__ classes use less memory but still allocate
   - Tuples benefit from size-specific freelists

5. FUNCTION CALLS:
   - Simple function calls have minimal allocation overhead
   - *args and **kwargs create tuples/dicts on each call
   - Lambda calls have similar overhead to regular functions
    """)


if __name__ == '__main__':
    main()
