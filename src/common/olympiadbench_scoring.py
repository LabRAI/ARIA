"""OlympiadBench mathematical-answer equivalence scoring.

Adapted from OpenBMB/OlympiadBench ``eval/auto_scoring_judge.py``:
https://github.com/OpenBMB/OlympiadBench

MIT License

Copyright (c) 2024 OpenBMB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import re

import sympy as sp
from sympy import Eq, Pow, simplify, sympify
from sympy.parsing.latex import parse_latex


class OlympiadBenchScorer:
    """Match the benchmark's public automatic scoring semantics."""

    def __init__(self) -> None:
        self.special_signal_map = {
            "\\left": "",
            "\\right": "",
            "∶": ":",
            "，": ",",
            "$": "",
            "\\approx": "=",
            "\\simeq": "=",
            "\\sim": "=",
            "^\\prime": "'",
            "^{\\prime}": "'",
            "^\\circ": "",
            "%": "",
        }
        self.pi = parse_latex("\\pi")
        self.precision = 1e-8

    def judge(
        self,
        reference: str,
        prediction: str,
        precision: float | list[float] = 1e-8,
    ) -> bool:
        try:
            reference, prediction = self._preprocess(reference, prediction)
        except (TypeError, ValueError):
            return False
        if reference == prediction and reference != "":
            return True

        reference = re.sub(r"[\u4e00-\u9fff]+", "", reference)
        prediction = re.sub(r"[\u4e00-\u9fff]+", "", prediction)
        references = self._expand_plus_minus(self._split_by_comma(reference))
        predictions = self._expand_plus_minus(self._split_by_comma(prediction))
        if len(references) != len(predictions):
            return False

        precisions = list(precision) if isinstance(precision, list) else [precision]
        if len(precisions) == 1:
            precisions *= len(references)
        if len(precisions) != len(references):
            return False

        remaining = list(predictions)
        for reference_item, item_precision in zip(references, precisions):
            self.precision = float(item_precision)
            for index, prediction_item in enumerate(remaining):
                if self._is_equal(reference_item, prediction_item):
                    remaining.pop(index)
                    break
            else:
                return False
        return not remaining

    @staticmethod
    def _split_by_comma(expression: str) -> list[str]:
        depth = 0
        values: list[str] = []
        start = 0
        for index, character in enumerate(expression):
            if character in "([":
                depth += 1
            elif character in ")]":
                depth -= 1
            elif character == "," and depth == 0:
                values.append(expression[start:index].strip())
                start = index + 1
        if start < len(expression):
            values.append(expression[start:].strip())
        return values

    @staticmethod
    def _expand_plus_minus(expressions: list[str]) -> list[str]:
        expanded: list[str] = []
        for expression in expressions:
            if "\\pm" in expression:
                expanded.extend(
                    [expression.replace("\\pm", "+"), expression.replace("\\pm", "-")]
                )
            else:
                expanded.append(expression)
        return expanded

    @staticmethod
    def _is_interval(expression: str) -> bool:
        return expression.startswith(("(", "[")) and expression.endswith((")", "]"))

    def _is_equal(self, reference: str, prediction: str) -> bool:
        if reference == prediction and reference != "":
            return True
        if self._is_interval(reference) and self._is_interval(prediction):
            try:
                return self._interval_equal(reference, prediction)
            except Exception:  # SymPy may raise LaTeXParsingError for malformed answers.
                return False
        try:
            if self._numerical_equal(reference, prediction):
                return True
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            if "=" not in reference or "=" not in prediction:
                if self._expression_equal(reference, prediction):
                    return True
        except Exception:  # SymPy parsers raise several backend-specific errors.
            pass
        try:
            return self._equation_equal(reference, prediction)
        except Exception:
            return False

    def _numerical_equal(self, reference: str, prediction: str) -> bool:
        reference_value = float(reference)
        prediction_value = float(prediction)
        return any(
            abs(candidate - prediction_value) <= self.precision * 1.01
            for candidate in (reference_value / 100, reference_value, reference_value * 100)
        )

    def _expression_equal(self, reference: str, prediction: str) -> bool:
        def expression_right_hand_side(value: str) -> str:
            return value.split("=", 1)[-1].strip()

        reference_sympy = sympify(parse_latex(expression_right_hand_side(reference)))
        prediction_sympy = sympify(parse_latex(expression_right_hand_side(prediction)))
        if reference_sympy == prediction_sympy:
            return True

        reference_sympy = reference_sympy.subs(self.pi, math.pi)
        prediction_sympy = prediction_sympy.subs(self.pi, math.pi)
        reference_has_symbol = bool(reference_sympy.has(sp.Symbol))
        prediction_has_symbol = bool(prediction_sympy.has(sp.Symbol))
        if reference_has_symbol != prediction_has_symbol:
            return False
        if not reference_has_symbol:
            if not (
                self._can_compute_power(reference_sympy)
                and self._can_compute_power(prediction_sympy)
            ):
                return False
            return abs(reference_sympy.evalf() - prediction_sympy.evalf()) <= self.precision * 1.01

        difference = simplify(reference_sympy - prediction_sympy)
        return bool(difference == 0 or abs(difference.evalf()) < 1e-3)

    @staticmethod
    def _equation_equal(reference: str, prediction: str) -> bool:
        def normalize_equation(value: str):
            left, right = value.split("=", 1)
            equation = Eq(parse_latex(left), parse_latex(right))
            return simplify(equation.lhs - equation.rhs)

        reference_sympy = normalize_equation(reference)
        prediction_sympy = normalize_equation(prediction)
        ratio = simplify(reference_sympy / prediction_sympy)
        inverse_ratio = simplify(prediction_sympy / reference_sympy)
        return bool(
            (ratio.is_Integer and ratio != 0)
            or (inverse_ratio.is_Integer and inverse_ratio != 0)
        )

    def _interval_equal(self, reference: str, prediction: str) -> bool:
        reference_intervals = reference.split("\\cup")
        prediction_intervals = prediction.split("\\cup")
        if len(reference_intervals) != len(prediction_intervals):
            return False
        for reference_interval, prediction_interval in zip(
            reference_intervals, prediction_intervals
        ):
            if (
                reference_interval[0] != prediction_interval[0]
                or reference_interval[-1] != prediction_interval[-1]
            ):
                return False
            reference_values = reference_interval.strip("[]()").split(",")
            prediction_values = prediction_interval.strip("[]()").split(",")
            if len(reference_values) != len(prediction_values):
                return False
            if not all(
                self._expression_equal(reference_value, prediction_value)
                for reference_value, prediction_value in zip(
                    reference_values, prediction_values
                )
            ):
                return False
        return True

    def _preprocess(self, reference: str, prediction: str) -> tuple[str, str]:
        return self._clean_expression(reference), self._clean_expression(prediction)

    def _clean_expression(self, expression: str) -> str:
        expression = self._extract_boxed_or_last_line_math(str(expression))
        if "\\in " in expression:
            expression = expression.split("\\in ", 1)[1]
        for signal, replacement in self.special_signal_map.items():
            expression = expression.replace(signal, replacement)
        expression = expression.strip("\n$,.:;^_=+`!@#$%^&*~，。")
        return re.sub(r"\\(?:mathrm|mathbf)\{~?([^}]*)\}", r"\1", expression)

    @staticmethod
    def _extract_boxed_or_last_line_math(expression: str) -> str:
        boxed: list[str] = []
        for match in re.finditer(r"\\boxed\{", expression):
            start = match.end()
            end = start
            depth = 1
            while depth > 0 and end < len(expression):
                if expression[end] == "{":
                    depth += 1
                elif expression[end] == "}":
                    depth -= 1
                end += 1
            if depth != 0:
                raise ValueError("mismatched braces in boxed OlympiadBench answer")
            boxed.append(expression[start : end - 1])
        if boxed:
            return ",".join(boxed)

        last_line = expression.strip().splitlines()[-1] if expression.strip() else ""
        math_fragments = re.findall(r"\$(.*?)\$", last_line)
        return ",".join(math_fragments) if math_fragments else expression

    @staticmethod
    def _can_compute_power(expression) -> bool:
        if not isinstance(expression, Pow):
            return True
        base, exponent = expression.as_base_exp()
        return bool(base.is_number and exponent.is_number and abs(exponent.evalf()) <= 1000)
