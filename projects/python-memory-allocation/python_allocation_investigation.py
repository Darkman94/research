#!/usr/bin/env python3
"""
Investigation of Python memory allocation rates for common operations.

This script measures how frequently different Python operations trigger memory allocations
using Python's tracemalloc module.
"""

import tracemalloc
import sys
from typing import Callable, Any, Dict, List, Tuple
import gc


class AllocationTracker:
    """Tracks memory allocations for a given operation."""

    def __init__(self, iterations: int = 100000):
        self.iterations = iterations
        self.results: List[Dict[str, Any]] = []

    def measure(self, name: str, operation: Callable, setup: Callable = None) -> Dict[str, Any]:
        """
        Measure allocations for a given operation.

        Args:
            name: Name of the operation
            operation: Function to test (takes iteration index as argument)
            setup: Optional setup function to run before measurement

        Returns:
            Dictionary with measurement results
        """
        # Force garbage collection before measurement
        gc.collect()

        # Setup if needed
        if setup:
            setup()

        # Start tracking
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        # Run the operation
        for i in range(self.iterations):
            operation(i)

        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate statistics
        stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_allocated = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        total_blocks = sum(stat.count_diff for stat in stats if stat.count_diff > 0)

        result = {
            'name': name,
            'iterations': self.iterations,
            'total_bytes_allocated': total_allocated,
            'total_allocations': total_blocks,
            'bytes_per_iteration': total_allocated / self.iterations,
            'allocations_per_iteration': total_blocks / self.iterations,
        }

        self.results.append(result)
        return result

    def print_result(self, result: Dict[str, Any]):
        """Print a single result in a formatted way."""
        print(f"\n{'=' * 70}")
        print(f"Operation: {result['name']}")
        print(f"Iterations: {result['iterations']:,}")
        print(f"Total bytes allocated: {result['total_bytes_allocated']:,}")
        print(f"Total allocations: {result['total_allocations']:,}")
        print(f"Bytes per iteration: {result['bytes_per_iteration']:.2f}")
        print(f"Allocations per iteration: {result['allocations_per_iteration']:.4f}")
        print(f"{'=' * 70}")

    def print_summary(self):
        """Print a summary table of all results."""
        print("\n" + "=" * 100)
        print(f"{'Operation':<40} {'Allocs/Iter':>15} {'Bytes/Iter':>15} {'Total Allocs':>15}")
        print("=" * 100)

        for result in sorted(self.results, key=lambda x: x['allocations_per_iteration'], reverse=True):
            print(f"{result['name']:<40} {result['allocations_per_iteration']:>15.4f} "
                  f"{result['bytes_per_iteration']:>15.2f} {result['total_allocations']:>15,}")

        print("=" * 100)


def test_integer_operations(tracker: AllocationTracker):
    """Test integer arithmetic operations."""
    print("\n### INTEGER OPERATIONS ###")

    # Simple addition
    x = 0
    tracker.measure(
        "Integer addition (small numbers)",
        lambda i: x + 1
    )

    # Addition with result storage
    result = [0]
    tracker.measure(
        "Integer addition with assignment",
        lambda i: result.__setitem__(0, result[0] + 1)
    )

    # Large number addition
    large = 10**100
    tracker.measure(
        "Large integer addition (>256)",
        lambda i: large + 1
    )

    # Integer in small range (-5 to 256, pre-allocated)
    tracker.measure(
        "Small integer creation (0-255)",
        lambda i: i % 256
    )

    # Integer outside pre-allocated range
    tracker.measure(
        "Large integer creation (>256)",
        lambda i: i + 1000
    )


def test_string_operations(tracker: AllocationTracker):
    """Test string operations."""
    print("\n### STRING OPERATIONS ###")

    # String concatenation
    s = "hello"
    tracker.measure(
        "String concatenation (small)",
        lambda i: s + "world"
    )

    # String formatting (f-string)
    tracker.measure(
        "String f-string formatting",
        lambda i: f"iteration {i}"
    )

    # String formatting (format method)
    tracker.measure(
        "String .format() method",
        lambda i: "iteration {}".format(i)
    )

    # String formatting (% operator)
    tracker.measure(
        "String % formatting",
        lambda i: "iteration %d" % i
    )

    # String slicing
    long_str = "a" * 1000
    tracker.measure(
        "String slicing",
        lambda i: long_str[10:20]
    )

    # String repetition
    tracker.measure(
        "String repetition",
        lambda i: "x" * 10
    )


def test_list_operations(tracker: AllocationTracker):
    """Test list operations."""
    print("\n### LIST OPERATIONS ###")

    # List append (pre-allocated list)
    lst = []
    tracker.measure(
        "List append (growing)",
        lambda i: lst.append(i)
    )

    # List comprehension
    tracker.measure(
        "List comprehension (10 items)",
        lambda i: [x for x in range(10)]
    )

    # List concatenation
    l1 = [1, 2, 3]
    tracker.measure(
        "List concatenation",
        lambda i: l1 + [4, 5, 6]
    )

    # List slicing
    big_list = list(range(1000))
    tracker.measure(
        "List slicing",
        lambda i: big_list[10:20]
    )

    # List creation with literals
    tracker.measure(
        "List literal creation",
        lambda i: [1, 2, 3, 4, 5]
    )


def test_dict_operations(tracker: AllocationTracker):
    """Test dictionary operations."""
    print("\n### DICTIONARY OPERATIONS ###")

    # Dict creation
    tracker.measure(
        "Dict literal creation",
        lambda i: {'a': 1, 'b': 2, 'c': 3}
    )

    # Dict insertion
    d = {}
    tracker.measure(
        "Dict insertion (growing)",
        lambda i: d.__setitem__(i, i)
    )

    # Dict comprehension
    tracker.measure(
        "Dict comprehension (10 items)",
        lambda i: {x: x*2 for x in range(10)}
    )

    # Dict lookup
    lookup_dict = {i: i*2 for i in range(1000)}
    tracker.measure(
        "Dict lookup",
        lambda i: lookup_dict.get(i % 1000)
    )


def test_function_calls(tracker: AllocationTracker):
    """Test function call overhead."""
    print("\n### FUNCTION CALLS ###")

    # Simple function call
    def simple_func(x):
        return x + 1

    tracker.measure(
        "Simple function call",
        lambda i: simple_func(i)
    )

    # Function with multiple args
    def multi_arg_func(a, b, c):
        return a + b + c

    tracker.measure(
        "Function call (3 args)",
        lambda i: multi_arg_func(i, i+1, i+2)
    )

    # Lambda function
    lam = lambda x: x + 1
    tracker.measure(
        "Lambda call",
        lambda i: lam(i)
    )

    # Method call
    class SimpleClass:
        def method(self, x):
            return x + 1

    obj = SimpleClass()
    tracker.measure(
        "Method call",
        lambda i: obj.method(i)
    )


def test_object_creation(tracker: AllocationTracker):
    """Test object creation."""
    print("\n### OBJECT CREATION ###")

    # Simple class instance
    class Point:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    tracker.measure(
        "Simple object creation",
        lambda i: Point(i, i+1)
    )

    # Class with __slots__
    class PointSlots:
        __slots__ = ['x', 'y']
        def __init__(self, x, y):
            self.x = x
            self.y = y

    tracker.measure(
        "Object creation (with __slots__)",
        lambda i: PointSlots(i, i+1)
    )

    # Tuple creation
    tracker.measure(
        "Tuple creation",
        lambda i: (i, i+1, i+2)
    )

    # Set creation
    tracker.measure(
        "Set creation",
        lambda i: {1, 2, 3, 4, 5}
    )


def test_control_flow(tracker: AllocationTracker):
    """Test control flow operations."""
    print("\n### CONTROL FLOW ###")

    # If statement
    tracker.measure(
        "If statement evaluation",
        lambda i: 1 if i % 2 == 0 else 0
    )

    # Generator expression (lazy)
    tracker.measure(
        "Generator expression creation",
        lambda i: (x for x in range(10))
    )

    # Range object
    tracker.measure(
        "Range object creation",
        lambda i: range(10)
    )


def main():
    """Run all allocation tests."""
    print("Python Memory Allocation Investigation")
    print("=" * 100)
    print(f"Python version: {sys.version}")
    print(f"Test iterations: 100,000 per operation")

    tracker = AllocationTracker(iterations=100000)

    # Run all test categories
    test_integer_operations(tracker)
    test_string_operations(tracker)
    test_list_operations(tracker)
    test_dict_operations(tracker)
    test_function_calls(tracker)
    test_object_creation(tracker)
    test_control_flow(tracker)

    # Print summary
    tracker.print_summary()

    # Additional analysis
    print("\n### KEY FINDINGS ###")

    # Find operations with < 0.01 allocations per iteration
    low_alloc = [r for r in tracker.results if r['allocations_per_iteration'] < 0.01]
    print(f"\nOperations with < 0.01 allocations/iteration ({len(low_alloc)} total):")
    for r in low_alloc:
        print(f"  - {r['name']}: {r['allocations_per_iteration']:.6f}")

    # Find operations with > 1 allocation per iteration
    high_alloc = [r for r in tracker.results if r['allocations_per_iteration'] > 1.0]
    print(f"\nOperations with > 1 allocation/iteration ({len(high_alloc)} total):")
    for r in high_alloc:
        print(f"  - {r['name']}: {r['allocations_per_iteration']:.2f}")

    # Save results to file
    with open('/home/user/research/allocation_results.txt', 'w') as f:
        f.write("Python Memory Allocation Investigation Results\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Iterations per test: {tracker.iterations:,}\n\n")

        f.write(f"{'Operation':<40} {'Allocs/Iter':>15} {'Bytes/Iter':>15} {'Total Allocs':>15}\n")
        f.write("=" * 100 + "\n")

        for result in sorted(tracker.results, key=lambda x: x['allocations_per_iteration'], reverse=True):
            f.write(f"{result['name']:<40} {result['allocations_per_iteration']:>15.4f} "
                   f"{result['bytes_per_iteration']:>15.2f} {result['total_allocations']:>15,}\n")

    print("\nResults saved to allocation_results.txt")


if __name__ == '__main__':
    main()
