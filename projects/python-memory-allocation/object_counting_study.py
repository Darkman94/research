#!/usr/bin/env python3
"""
Study Python allocations by counting actual object creations.

This uses gc and id() tracking to detect when new objects are created vs reused.
"""

import sys
import gc
from typing import Set, List, Dict, Any


def count_new_objects(operation, iterations: int = 10000) -> Dict[str, Any]:
    """
    Count how many new objects are created during an operation.

    This works by tracking object IDs before and after, though it's
    not perfect due to object reuse and garbage collection.
    """
    # Force collection and get baseline
    gc.collect()
    gc.disable()  # Disable during measurement

    before_objects = set(id(obj) for obj in gc.get_objects())

    # Run operation
    operation(iterations)

    after_objects = set(id(obj) for obj in gc.get_objects())

    gc.enable()

    new_objects = len(after_objects - before_objects)

    return {
        'new_objects': new_objects,
        'objects_per_iteration': new_objects / iterations if iterations > 0 else 0
    }


def test_with_id_tracking():
    """
    Test allocation patterns by tracking object identity.

    This reveals whether Python reuses objects or creates new ones.
    """
    print("=" * 80)
    print("OBJECT IDENTITY TRACKING STUDY")
    print("=" * 80)
    print()

    # Test 1: Integer identity reuse
    print("1. INTEGER IDENTITY PATTERNS")
    print("-" * 80)

    # Small integers (pre-allocated)
    small_ints = [i for i in range(10)]
    small_ints_again = [i for i in range(10)]
    same_ids = sum(1 for i in range(10) if id(small_ints[i]) == id(small_ints_again[i]))
    print(f"Small integers (0-9): {same_ids}/10 have same ID (reused)")

    # Medium integers (in cache range)
    med_ints = [i for i in range(250, 260)]
    med_ints_again = [i for i in range(250, 260)]
    same_ids = sum(1 for i in range(10) if id(med_ints[i]) == id(med_ints_again[i]))
    print(f"Medium integers (250-259): {same_ids}/10 have same ID (reused)")

    # Large integers (outside cache)
    large_ints = [i for i in range(1000, 1010)]
    large_ints_again = [i for i in range(1000, 1010)]
    same_ids = sum(1 for i in range(10) if id(large_ints[i]) == id(large_ints_again[i]))
    print(f"Large integers (1000-1009): {same_ids}/10 have same ID (reused)")

    # Test 2: String identity (interning)
    print("\n2. STRING IDENTITY PATTERNS")
    print("-" * 80)

    # Literal strings (interned)
    s1 = "hello"
    s2 = "hello"
    print(f"String literals: id('{s1}') == id('{s2}'): {id(s1) == id(s2)}")

    # Constructed strings (not always interned)
    s3 = "hel" + "lo"
    print(f"Constructed string: id('hel' + 'lo') == id('hello'): {id(s3) == id(s1)}")

    # Runtime concatenation
    prefix = "hel"
    s4 = prefix + "lo"
    print(f"Runtime concat: same as literal: {id(s4) == id(s1)}")

    # Interned strings
    s5 = sys.intern("hello")
    print(f"sys.intern('hello'): same as literal: {id(s5) == id(s1)}")

    # Test 3: Container reuse
    print("\n3. CONTAINER IDENTITY PATTERNS")
    print("-" * 80)

    # Empty tuple (singleton)
    t1 = ()
    t2 = ()
    print(f"Empty tuples: id(()) == id(()): {id(t1) == id(t2)}")

    # Small tuples
    t3 = (1, 2)
    t4 = (1, 2)
    print(f"Tuples (1, 2): same ID: {id(t3) == id(t4)}")

    # Empty lists (not singleton)
    l1 = []
    l2 = []
    print(f"Empty lists: id([]) == id([]): {id(l1) == id(l2)}")

    # Test 4: Boolean and None (singletons)
    print("\n4. SINGLETON OBJECTS")
    print("-" * 80)
    print(f"True is True: {True is True}")
    print(f"False is False: {False is False}")
    print(f"None is None: {None is None}")
    print(f"All True values have same ID: {id(True) == id(1 == 1)}")


def analyze_operation_with_disassembly():
    """
    Analyze what operations actually do at the bytecode level.
    """
    import dis

    print("\n" + "=" * 80)
    print("BYTECODE ANALYSIS")
    print("=" * 80)

    print("\n1. Simple integer addition:")
    print("-" * 80)
    def int_add():
        x = 1
        y = x + 1
        return y

    dis.dis(int_add)

    print("\n2. String concatenation:")
    print("-" * 80)
    def str_concat():
        x = "hello"
        y = x + "world"
        return y

    dis.dis(str_concat)

    print("\n3. List append:")
    print("-" * 80)
    def list_append_op():
        x = []
        x.append(1)
        return x

    dis.dis(list_append_op)

    print("\n4. List comprehension:")
    print("-" * 80)
    def list_comp():
        return [i for i in range(10)]

    dis.dis(list_comp)


def measure_operation_speed():
    """
    Time various operations to show relative performance.
    """
    import timeit

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON (microseconds per operation)")
    print("=" * 80)

    operations = {
        "Integer addition (small)": "x = 1 + 1",
        "Integer addition (large)": "x = 1000000 + 1",
        "String concatenation": "x = 'hello' + 'world'",
        "F-string formatting": "x = f'value: {42}'",
        "List append": "lst.append(1)",
        "Dict insert": "d[1] = 1",
        "Tuple creation": "t = (1, 2, 3)",
        "List literal": "x = [1, 2, 3]",
        "Function call": "func(1)",
        "Method call": "obj.method(1)",
    }

    setup_code = """
lst = []
d = {}
def func(x): return x + 1
class C:
    def method(self, x): return x + 1
obj = C()
"""

    print(f"\n{'Operation':<30} {'Time (μs)':>15} {'Relative':>10}")
    print("-" * 60)

    results = []
    for name, code in operations.items():
        time = timeit.timeit(code, setup=setup_code, number=100000) / 100000 * 1_000_000
        results.append((name, time))

    # Sort by time
    results.sort(key=lambda x: x[1])
    baseline = results[0][1]

    for name, time in results:
        relative = time / baseline
        print(f"{name:<30} {time:>15.4f} {relative:>10.2f}x")


def main():
    print("COMPREHENSIVE PYTHON ALLOCATION PATTERNS")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print()

    test_with_id_tracking()
    analyze_operation_with_disassembly()
    measure_operation_speed()

    print("\n" + "=" * 80)
    print("SUMMARY OF FINDINGS")
    print("=" * 80)
    print("""
Python employs several strategies to minimize allocations:

1. OBJECT CACHING & SINGLETONS:
   - Integers from -5 to 256 are pre-allocated (single shared instances)
   - Empty tuples () are singletons
   - True, False, and None are singletons
   - Small strings may be interned automatically

2. FREELISTS (Not directly observable via Python code):
   - Integers, floats, tuples, and frames use freelists
   - Deallocated objects go to freelist for quick reuse
   - Avoids malloc/free calls to system allocator

3. MEMORY POOLING:
   - Python's allocator (pymalloc) manages small objects (<512 bytes)
   - Uses arenas, pools, and blocks for efficient allocation
   - Reduces fragmentation and system allocator calls

4. OPERATIONS THAT ALWAYS ALLOCATE:
   - String operations (strings are immutable)
   - List/dict growth beyond current capacity
   - New object instantiation
   - Function calls with *args or **kwargs

5. OPERATIONS THAT RARELY ALLOCATE:
   - Small integer arithmetic (uses cache + freelist)
   - Boolean operations (singletons)
   - Empty container checks
   - Simple function calls with regular args

The blog post's finding that "Python allocates very often" is accurate when
looking at internal allocation function calls, but many of these allocations
are satisfied by freelists rather than actual malloc() calls. From a performance
perspective, freelist allocations are very fast (pointer manipulation).
    """)


if __name__ == '__main__':
    main()
