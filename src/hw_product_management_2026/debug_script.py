"""Debug script for testing linter and type checker."""

from collections.abc import Sequence


def summarize(numbers: Sequence[float]) -> dict[str, float]:
    """Return basic statistics for a sequence of numbers."""
    if not numbers:
        return {"count": 0.0, "sum": 0.0, "mean": 0.0}

    total = sum(numbers)
    count = len(numbers)
    return {
        "count": float(count),
        "sum": float(total),
        "mean": total / count,
    }


def main() -> None:
    """Run a demo summary."""
    data = [1.0, 2.5, 3.7, 4.2]
    result = summarize(data)
    print(result)


if __name__ == "__main__":
    main()
