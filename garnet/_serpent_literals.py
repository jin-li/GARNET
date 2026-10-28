"""Internal literal grammar shared by native Serpent result readers.

This handles indexed result assignments, quoted strings, and numeric vectors;
it never evaluates MATLAB. Schema-specific finiteness and shape validation
belongs to each reader. Micro-depletion matrices use a separate row grammar.
"""

import re

import numpy as np


_ASSIGNMENT = re.compile(
    r"^([A-Z][A-Z0-9_]*)\s*\(\s*idx\s*,\s*(1|\[\s*1\s*:\s*(\d+)\s*\])"
    r"\s*\)\s*=\s*(.*?)\s*$", re.S)
_NUMBER = re.compile(
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?|inf|nan)$", re.I)
_COUNTER = re.compile(r"^if\s*\(\s*exist\s*\(\s*'idx'\s*,\s*'var'\s*\)\s*\)$")


def _statements(text):
    """Split MATLAB data statements; honor quoted strings and percent comments."""
    current, quoted, comment, brackets = [], False, False, 0
    i = 0
    while i < len(text):
        c = text[i]
        if comment:
            if c == "\n":
                comment = False
                current.append(c)
        elif quoted:
            current.append(c)
            if c == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    current.append("'")
                    i += 1
                else:
                    quoted = False
        elif c == "%":
            comment = True
        elif c == "'":
            quoted = True
            current.append(c)
        elif c == ";" and brackets == 0:
            yield "".join(current).strip()
            current = []
        else:
            brackets += (c == "[") - (c == "]")
            current.append(c)
        i += 1
    if quoted or brackets != 0:
        raise ValueError("unterminated string or numeric vector")
    if "".join(current).strip():
        raise ValueError("unterminated statement (missing semicolon)")


def _literal(rhs):
    if re.fullmatch(r"'(?:[^']|'')*'", rhs, re.S):
        return rhs[1:-1].replace("''", "'")
    if rhs.startswith("[") and rhs.endswith("]"):
        rhs = rhs[1:-1].strip()
    tokens = rhs.replace(",", " ").split()
    if not tokens or any(not _NUMBER.fullmatch(t) for t in tokens):
        raise ValueError("expected a numeric literal/vector or quoted string; expressions are unsupported")
    return np.array([float(t.replace("D", "E").replace("d", "e")) for t in tokens])

