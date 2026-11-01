"""Unit conversion tool for agents."""

from __future__ import annotations


# Conversion factors to base units.
_LENGTH_TO_METER = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "yd": 0.9144,
    "yard": 0.9144,
    "yards": 0.9144,
    "mi": 1609.344,
    "mile": 1609.344,
    "miles": 1609.344,
}

_WEIGHT_TO_KG = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 0.001,
    "grams": 0.001,
    "mg": 0.000001,
    "milligram": 0.000001,
    "milligrams": 0.000001,
    "lb": 0.453592,
    "lbs": 0.453592,
    "pound": 0.453592,
    "pounds": 0.453592,
    "oz": 0.0283495,
    "ounce": 0.0283495,
    "ounces": 0.0283495,
}


def unit_conversion_tool(
    value: float,
    from_unit: str,
    to_unit: str,
    category: str = "length",
) -> str:
    """Convert a value between units.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit name (e.g., "m", "km", "ft").
        to_unit: Target unit name (e.g., "cm", "mi", "in").
        category: Conversion category ("length" or "weight").

    Returns:
        A human-readable conversion result string.

    Raises:
        ValueError: If units are unknown or category is unsupported.
    """
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    if category == "length":
        table = _LENGTH_TO_METER
        base_name = "meters"
    elif category == "weight":
        table = _WEIGHT_TO_KG
        base_name = "kilograms"
    else:
        raise ValueError(f"unsupported category: {category!r}")

    if from_unit not in table:
        raise ValueError(f"unknown from_unit: {from_unit!r}")
    if to_unit not in table:
        raise ValueError(f"unknown to_unit: {to_unit!r}")

    # Convert to base unit, then to target unit.
    base_value = value * table[from_unit]
    result = base_value / table[to_unit]

    return f"{value} {from_unit} = {result:.6g} {to_unit}"