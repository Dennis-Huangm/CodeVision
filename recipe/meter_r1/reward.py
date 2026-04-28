"""MeterRead-R1 reward scorer for CodeVision/VERL."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(
    r"(?<![\w.,])([+\-−]?(?:(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?)(?![\d.,])"
)

UNIT_ALIASES = {
    "v": "V",
    "dcv": "V",
    "acv": "V",
    "vdc": "V",
    "vac": "V",
    "volt": "V",
    "volts": "V",
    "a": "A",
    "amp": "A",
    "amps": "A",
    "ampere": "A",
    "amperes": "A",
    "ma": "mA",
    "dcma": "mA",
    "acma": "mA",
    "milliamp": "mA",
    "milliamps": "mA",
    "ua": "μA",
    "µa": "μA",
    "μa": "μA",
    "microamp": "μA",
    "microamps": "μA",
    "ohm": "Ω",
    "ohms": "Ω",
    "ω": "Ω",
    "c": "°C",
    "°c": "°C",
    "degc": "°C",
    "degrees c": "°C",
    "degree c": "°C",
    "degreescelsius": "°C",
    "degreecelsius": "°C",
    "degrees celsius": "°C",
    "degree celsius": "°C",
    "celsius": "°C",
    "f": "°F",
    "°f": "°F",
    "degf": "°F",
    "degreesfahrenheit": "°F",
    "degreefahrenheit": "°F",
    "degrees fahrenheit": "°F",
    "degree fahrenheit": "°F",
    "fahrenheit": "°F",
}

UNIT_PREFIXES = tuple(
    sorted(
        (
            (alias.replace("µ", "μ").lower().replace(" ", ""), normalized)
            for alias, normalized in UNIT_ALIASES.items()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)


def _is_unit_boundary(char: str) -> bool:
    return char.isspace() or unicodedata.category(char).startswith(("P", "S"))


def _parse_number(value_text: str) -> float | None:
    try:
        value = float(value_text.replace("−", "-").replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _match_unit_prefix(text: str) -> str | None:
    for alias_key, normalized in UNIT_PREFIXES:
        position = 0
        key_position = 0
        while position < len(text) and text[position].isspace():
            position += 1
        while key_position < len(alias_key):
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text):
                break
            char = text[position].replace("µ", "μ").lower()
            if char != alias_key[key_position]:
                break
            position += 1
            key_position += 1
        if key_position != len(alias_key):
            continue
        if position < len(text) and not _is_unit_boundary(text[position]):
            continue
        return normalized
    return None


def _last_prediction(
    text: str, *, prefer_unit: bool = False
) -> tuple[float, str | None] | None:
    prediction: tuple[float, str | None] | None = None
    unit_prediction: tuple[float, str | None] | None = None
    for match in NUMBER_RE.finditer(text):
        value = _parse_number(match.group(1))
        if value is None:
            continue
        unit = _match_unit_prefix(text[match.end() :])
        prediction = (value, unit)
        if unit is not None:
            unit_prediction = prediction
    if prefer_unit and unit_prediction is not None:
        return unit_prediction
    return prediction


def extract_answer_text(solution_str: str) -> tuple[str | None, bool]:
    if not solution_str:
        return None, False
    cleaned = re.sub(
        r"<tool_response>.*?</tool_response>", "", solution_str, flags=re.DOTALL
    )
    matches = ANSWER_RE.findall(cleaned)
    if matches:
        return matches[-1].strip(), True

    if _last_prediction(cleaned) is None:
        return None, False
    return cleaned, True


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    compact = " ".join(unit.replace("µ", "μ").strip().split())
    if not compact:
        return None
    key = compact.lower().replace(" ", "")
    if key in UNIT_ALIASES:
        return UNIT_ALIASES[key]
    spaced_key = compact.lower()
    if spaced_key in UNIT_ALIASES:
        return UNIT_ALIASES[spaced_key]
    return compact


def parse_prediction(
    answer_text: str | None, *, prefer_unit: bool = False
) -> tuple[float | None, str | None]:
    if answer_text is None:
        return None, None
    prediction = _last_prediction(answer_text, prefer_unit=prefer_unit)
    if prediction is None:
        return None, None
    return prediction


def _zero_result(
    *,
    pred_value: float | None,
    pred_unit: str | None,
    format_ok: bool,
    unit_match: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "score": 0.0,
        "pred_value": pred_value,
        "pred_unit": pred_unit,
        "unit_match": unit_match,
        "error_ticks": None,
        "exact": False,
        "within_1_tick": False,
        "format_ok": format_ok,
    }
    if error is not None:
        result["error"] = error
    return result


def _read_ground_truth(ground_truth: dict[str, Any]) -> tuple[float, str | None, float]:
    gold_answer = ground_truth["gold_answer"]
    y_star = float(gold_answer["y_star"])
    unit = normalize_unit(gold_answer.get("unit"))
    tick_size = float(ground_truth["scale_info"]["tick_size"])
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    return y_star, unit, tick_size


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del data_source, extra_info
    answer_text, format_ok = extract_answer_text(solution_str)
    try:
        y_star, gt_unit, tick_size = _read_ground_truth(ground_truth)
    except (KeyError, TypeError, ValueError) as exc:
        pred_value, pred_unit = parse_prediction(answer_text)
        return _zero_result(
            pred_value=pred_value,
            pred_unit=pred_unit,
            format_ok=format_ok,
            error=f"invalid_ground_truth: {exc}",
        )

    pred_value, pred_unit = parse_prediction(
        answer_text, prefer_unit=format_ok and gt_unit is not None
    )
    if pred_value is None:
        return _zero_result(
            pred_value=None,
            pred_unit=None,
            format_ok=False,
            error="missing_numeric_prediction",
        )

    unit_match = gt_unit is None or pred_unit == gt_unit
    if not unit_match:
        return _zero_result(
            pred_value=pred_value,
            pred_unit=pred_unit,
            format_ok=format_ok,
            unit_match=False,
            error="unit_mismatch",
        )

    error_ticks = abs(pred_value - y_star) / tick_size
    score = max(0.0, 1.0 - 0.5 * error_ticks)
    return {
        "score": score,
        "pred_value": pred_value,
        "pred_unit": pred_unit,
        "unit_match": True,
        "error_ticks": error_ticks,
        "exact": error_ticks == 0.0,
        "within_1_tick": error_ticks <= 1.0,
        "format_ok": format_ok,
    }
