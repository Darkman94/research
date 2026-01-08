#!/usr/bin/env python3
"""
Create visualizations of Python allocation patterns.

Generates ASCII charts and formatted tables for the research findings.
"""


def create_ascii_bar_chart(data: list, title: str, max_width: int = 60):
    """Create an ASCII bar chart."""
    print(f"\n{title}")
    print("=" * 80)

    if not data:
        print("No data to display")
        return

    # Find max value for scaling
    max_val = max(val for _, val in data)
    if max_val == 0:
        max_val = 1

    for label, value in data:
        # Calculate bar width
        bar_width = int((value / max_val) * max_width)
        bar = "█" * bar_width

        # Format the output
        print(f"{label:<35} {bar} {value:.4f}")

    print()


def visualize_allocation_patterns():
    """Create visualizations of allocation patterns."""

    # Integer identity reuse rates
    integer_data = [
        ("Small (0-9)", 10/10),
        ("Medium (250-259)", 7/10),
        ("Large (1000+)", 0/10),
    ]

    create_ascii_bar_chart(
        integer_data,
        "INTEGER OBJECT REUSE RATE (higher is better)"
    )

    # Performance comparison (microseconds)
    perf_data = [
        ("String concat", 0.0079),
        ("Int add (large)", 0.0079),
        ("Int add (small)", 0.0080),
        ("Tuple creation", 0.0080),
        ("Dict insert", 0.0208),
        ("List append", 0.0280),
        ("List literal", 0.0316),
        ("Function call", 0.0345),
        ("Method call", 0.0367),
        ("F-string format", 0.0648),
    ]

    create_ascii_bar_chart(
        perf_data,
        "OPERATION PERFORMANCE (μs per operation, lower is better)"
    )

    # Relative performance
    baseline = 0.0079
    relative_data = [
        (label, value / baseline) for label, value in perf_data
    ]

    create_ascii_bar_chart(
        relative_data,
        "RELATIVE PERFORMANCE (normalized to fastest)"
    )

    # Allocation intensity (conceptual)
    alloc_intensity = [
        ("Singleton access", 0.00),
        ("Small int arithmetic", 0.01),
        ("Tuple creation", 0.02),
        ("Function call", 0.03),
        ("Dict lookup", 0.05),
        ("String concat", 0.80),
        ("Object creation", 0.85),
        ("F-string format", 0.90),
        ("List append (growth)", 0.99),
        ("Dict insert (growth)", 0.99),
    ]

    create_ascii_bar_chart(
        alloc_intensity,
        "ALLOCATION INTENSITY (allocations per iteration)"
    )


def create_comparison_table():
    """Create comparison tables."""

    print("\n" + "=" * 80)
    print("OPTIMIZATION STRATEGIES SUMMARY")
    print("=" * 80)

    strategies = [
        ("Strategy", "Applies To", "Benefit"),
        ("-" * 20, "-" * 30, "-" * 25),
        ("Object Caching", "Small ints, None, True/False", "Zero allocations"),
        ("String Interning", "String literals", "Memory sharing"),
        ("Freelists", "Int, float, tuple, frame", "Fast reuse, no malloc"),
        ("Pymalloc", "Objects < 512 bytes", "Reduced fragmentation"),
        ("Over-allocation", "Growing lists/dicts", "Amortized O(1) growth"),
        ("Bytecode Optimization", "Common patterns", "Reduced overhead"),
    ]

    for row in strategies:
        print(f"{row[0]:<22} {row[1]:<32} {row[2]:<25}")

    print("\n" + "=" * 80)
    print("WHEN TO WORRY ABOUT ALLOCATIONS")
    print("=" * 80)

    print("""
HIGH IMPACT (consider optimizing):
  ✓ Tight loops with string concatenation
  ✓ Building large lists incrementally (use list comp or pre-allocate)
  ✓ Creating many short-lived objects
  ✓ Heavy use of *args/**kwargs in hot paths

LOW IMPACT (usually not worth optimizing):
  ✓ Small integer arithmetic
  ✓ Dictionary lookups
  ✓ Simple function calls
  ✓ Boolean operations
  ✓ Working with cached values

MEASUREMENT TIP:
  Use profiling tools (cProfile, py-spy, memray) to identify actual
  bottlenecks before optimizing. Premature optimization based on
  allocation counts alone can be misleading.
    """)


def create_freelist_visualization():
    """Visualize the freelist concept."""

    print("\n" + "=" * 80)
    print("FREELIST REUSE PATTERN (Conceptual)")
    print("=" * 80)

    print("""
Traditional Allocation (every operation calls malloc):
┌─────────────────────────────────────────────────────────────┐
│  Operation → malloc() → OS Allocator → Memory                │
│     (slow)      (syscall)                                    │
└─────────────────────────────────────────────────────────────┘

With Freelist (most operations reuse):
┌─────────────────────────────────────────────────────────────┐
│  Operation → Check Freelist → Found! → Reuse Object          │
│                 (fast)           ↓                           │
│                              Not found                       │
│                                 ↓                            │
│                             malloc()                         │
│                           (occasional)                       │
└─────────────────────────────────────────────────────────────┘

Example: Integer Operations in 100,000 iterations
┌──────────────────────────────────────────────────────────────┐
│  Allocation calls:    ~99,100                                │
│  Freelist hits:       ~99,193  (reused objects)             │
│  New malloc calls:    ~102     (only when freelist empty)   │
│                                                              │
│  Result: 99.9% of allocations avoid expensive malloc()      │
└──────────────────────────────────────────────────────────────┘
    """)


def main():
    """Generate all visualizations."""

    print("=" * 80)
    print("PYTHON MEMORY ALLOCATION PATTERNS - VISUALIZATIONS")
    print("=" * 80)

    visualize_allocation_patterns()
    create_comparison_table()
    create_freelist_visualization()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
While Python internally calls allocation functions frequently, sophisticated
optimizations (freelists, caching, pooling) mean that:

1. Most "allocations" are actually fast freelist reuses
2. Only a small fraction trigger expensive malloc() calls
3. Performance is generally excellent for typical workloads
4. Optimization efforts should focus on algorithmic improvements first

Focus on:
  • Using appropriate data structures
  • Avoiding unnecessary object creation in tight loops
  • Pre-allocating when size is known
  • Profiling before optimizing

Don't worry about:
  • Individual integer operations
  • Simple function calls
  • Using built-in types naturally
  • Small temporary objects
    """)

    print("\nFor more details, see README.md in this directory.")
    print("=" * 80)


if __name__ == '__main__':
    main()
