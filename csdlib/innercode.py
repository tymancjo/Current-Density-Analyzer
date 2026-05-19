"""This is the library set for the inner code operations"""

import math as _math
from csdlib import csdlib as csd

# ---------------------------------------------------------------------------
# Safe expression evaluator
# ---------------------------------------------------------------------------

_SAFE_GLOBALS = {
    "__builtins__": {},
    # trig
    "sin":   _math.sin,   "cos":   _math.cos,   "tan":   _math.tan,
    "asin":  _math.asin,  "acos":  _math.acos,  "atan":  _math.atan,
    "atan2": _math.atan2,
    # exponential / log
    "sqrt":  _math.sqrt,  "exp":   _math.exp,
    "log":   _math.log,   "log10": _math.log10,
    # rounding
    "abs":   abs,   "round": round,   "int": int,   "float": float,
    "min":   min,   "max":   max,   "ceil": _math.ceil,   "floor": _math.floor,
    # constants
    "pi":    _math.pi,    "e":     _math.e,
    # math module (for math.sin etc.)
    "math":  _math,
}


def _eval(expr, variables):
    """Evaluate an arithmetic expression with the current variable namespace.

    Supports all standard arithmetic operators (+, -, *, /, **, %) and the
    math functions listed in _SAFE_GLOBALS (sin, cos, sqrt, pi, …).
    Variable names defined with v() are available by name.
    """
    return float(eval(str(expr).strip(), _SAFE_GLOBALS, dict(variables)))


def _resolve_args(ar, variables):
    """Evaluate each argument string as an expression; return a list of floats.

    Arguments that cannot be evaluated are returned unchanged so that the
    downstream float() conversion will produce a readable error.
    """
    resolved = []
    for arg in ar:
        try:
            resolved.append(_eval(arg, variables))
        except Exception:
            resolved.append(arg)
    return resolved


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def addCircle(x0, y0, D1, Set, D2=0, Set2=0, draw=True, shift=(0, 0),
              XSecArray=None, dXmm=1):
    """Add a filled circle or annulus centred at (x0, y0) [mm].

    D1 – outer diameter [mm], Set  – phase/material ID for the outer ring.
    D2 – inner diameter [mm], Set2 – phase/material ID for the inner fill.
    Omit D2 / Set2 for a solid disc.
    """
    if draw:
        x0 = x0 - shift[0]
        y0 = y0 - shift[1]

        r1sq = (D1 / 2) ** 2
        r2sq = (D2 / 2) ** 2

        elementsInY = XSecArray.shape[0]
        elementsInX = XSecArray.shape[1]

        for x in range(elementsInX):
            for y in range(elementsInY):
                xmm = x * dXmm + dXmm / 2
                ymm = y * dXmm + dXmm / 2
                distSq = (xmm - x0) ** 2 + (ymm - y0) ** 2
                if distSq < r2sq:
                    XSecArray[y, x] = Set2
                elif distSq <= r1sq:
                    XSecArray[y, x] = Set

    x0 = x0 - D1 / 2
    y0 = y0 - D1 / 2
    xE = x0 + D1
    yE = y0 + D1
    return [x0, y0, xE, yE]


def addRect(x0, y0, W, H, Set, draw=True, shift=(0, 0), XSecArray=None, dXmm=1):
    """Add a filled rectangle with top-left corner at (x0, y0) [mm],
    width W and height H [mm], assigned to phase/material ID Set.
    """
    xE = x0 + W
    yE = y0 + H

    if draw:
        x0 = x0 - shift[0]
        y0 = y0 - shift[1]
        xE = x0 + W
        yE = y0 + H

        elementsInY = XSecArray.shape[0]
        elementsInX = XSecArray.shape[1]

        for x in range(elementsInX):
            for y in range(elementsInY):
                xmm = x * dXmm + dXmm / 2
                ymm = y * dXmm + dXmm / 2

                if (x0 <= xmm <= xE) and (y0 <= ymm <= yE):
                    XSecArray[y, x] = Set

    return [x0, y0, xE, yE]


def moveCells(phase, shift_X, shift_Y, XSecArray=None, dXmm=1):
    dX = int(shift_X / dXmm)
    dY = int(shift_Y / dXmm)
    csd.n_shiftPhase(phase, dX, dY, XSecArray)


def copyCells(phase, shift_X, shift_Y, XSecArray=None, dXmm=1):
    dX = int(shift_X / dXmm)
    dY = int(shift_Y / dXmm)
    csd.n_shiftPhase(phase, dX, dY, XSecArray, remain=phase)


# ---------------------------------------------------------------------------
# Loop unwinding
# ---------------------------------------------------------------------------

def codeLoops(input_text):
    """Unwind l(n)…break blocks into flat repeated code.

    Loops are processed from the innermost outward (scanned backwards).
    The body of each l(n) runs exactly n times in total.
    """
    commands = {
        'l': [None, [1]]
    }

    if 'break' in input_text:
        break_index = input_text.index('break')
        rest_of_text = input_text[break_index + 1:]
        input_text = input_text[:break_index]
    else:
        rest_of_text = []

    for line_nr, line in enumerate(reversed(input_text)):
        index_non_rev = len(input_text) - line_nr

        if len(line) > 3:
            command = line.split('(')[0].lower().strip()
            if command in commands and command == 'l':
                arguments = line[2:-1].split(',')
                if len(arguments) in commands[command][1]:
                    loops = int(arguments[0]) - 1  # -1 because body is already present once
                    loop_code = input_text[index_non_rev:]
                    input_text[index_non_rev - 1] = "\n"
                    for _ in range(loops):
                        input_text.extend(loop_code)
                    codeLoops(input_text)

    return input_text + rest_of_text


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def textToCode(input_text):
    """Parse inner-code text and return
    (codeSteps, currents, materials, custom_materials, analysis_params).

    codeSteps        – list of [function, [float_args], command_name]
    currents         – list of [phase_id, I_rms, phase_shift, extra_shift] or [..., Isc_kA]
    materials        – list of [phase_id, material_type_id_or_name]
    custom_materials – dict of {name: {sigma, alpha, mi_r, rho, cp, k}}
                       populated by defmat() commands in the .ic file
    analysis_params  – dict with any of: freq, length, temp, htc, cellsize
                       populated by the matching single-argument directives
    """

    # Analysis-parameter directives: each takes exactly one numeric argument.
    _ANALYSIS_KEYS = {'freq', 'length', 'temp', 'htc', 'cellsize'}

    commands = {
        'c':        [addCircle, [4, 6]],
        'r':        [addRect,   [5]],
        'v':        [None,      [2]],
        'a':        [None,      [2]],
        'l':        [None,      [1]],
        'mv':       [moveCells, [3]],
        'cp':       [copyCells, [3]],
        'current':  [None,      [4, 5]],  # 5th arg is optional Isc (kA) for force analysis
        'material': [None,      [2]],
        'defmat':   [None,      [4, 7]],
    }

    innerCodeSteps = []
    innerVariables = {}
    currents = []
    materials = []
    custom_materials = {}
    analysis_params  = {}

    input_text = codeLoops(input_text)

    for line_nr, line in enumerate(input_text):
        line = line.strip()
        if not line or line.startswith('#') or len(line) < 3:
            continue

        command = line.split('(')[0].lower().strip()

        if command not in commands and command not in _ANALYSIS_KEYS:
            continue

        # Extract argument string: everything between the first '(' and last ')'
        try:
            raw_args = line.split('(', 1)[1]
            last_paren = raw_args.rfind(')')
            if last_paren != -1:
                raw_args = raw_args[:last_paren]
        except IndexError:
            print(f"[ic line {line_nr + 1}] Cannot parse arguments in: '{line}'")
            continue

        if command == 'v':
            # v(name, expression)  — split only on the first comma so the
            # expression may itself contain commas (e.g. max(a,b))
            ar = [a.strip() for a in raw_args.split(',', 1)]
            if len(ar) in commands[command][1]:
                variable_name = ar[0]
                try:
                    innerVariables[variable_name] = _eval(ar[1], innerVariables)
                except Exception as e:
                    print(f"[ic line {line_nr + 1}] Error evaluating v({ar[0]}, {ar[1]}): {e}")

        elif command == 'a':
            # a(name, expression)  — adds the expression result to an existing variable
            ar = [a.strip() for a in raw_args.split(',', 1)]
            if len(ar) in commands[command][1]:
                variable_name = ar[0]
                if variable_name not in innerVariables:
                    print(f"[ic line {line_nr + 1}] Warning: variable '{variable_name}' "
                          f"used in a() before v() initialisation — skipping")
                    continue
                try:
                    delta = _eval(ar[1], innerVariables)
                    innerVariables[variable_name] += delta
                except Exception as e:
                    print(f"[ic line {line_nr + 1}] Error evaluating a({ar[0]}, {ar[1]}): {e}")

        elif command == 'current':
            ar = [a.strip() for a in raw_args.split(',')]
            resolved = _resolve_args(ar, innerVariables)
            if len(resolved) in commands[command][1]:
                currents.append(resolved)

        elif command == 'material':
            # Split on first comma only so phase_id expressions like ph+1 work.
            # The material reference (second arg) may be an integer index OR a
            # name defined by defmat(); _resolve_args returns it as a string
            # when it cannot be evaluated as a numeric expression.
            ar = [a.strip() for a in raw_args.split(',', 1)]
            if len(ar) == 2:
                phase_resolved = _resolve_args([ar[0]], innerVariables)
                mat_resolved   = _resolve_args([ar[1]], innerVariables)
                materials.append([phase_resolved[0], mat_resolved[0]])

        elif command == 'defmat':
            # defmat(name, sigma, alpha, mi_r)
            # defmat(name, sigma, alpha, mi_r, rho, cp, thermal_k)
            # First token is always a plain name string; rest are numeric expressions.
            parts = [a.strip() for a in raw_args.split(',', 1)]
            if len(parts) != 2:
                print(f"[ic line {line_nr + 1}] defmat requires at least 4 arguments")
                continue
            mat_name = parts[0]
            numeric_parts = [a.strip() for a in parts[1].split(',')]
            resolved = _resolve_args(numeric_parts, innerVariables)
            if len(resolved) == 3:
                # sigma, alpha, mi_r
                try:
                    custom_materials[mat_name] = {
                        'sigma': float(resolved[0]),
                        'alpha': float(resolved[1]),
                        'mi_r':  float(resolved[2]),
                        'rho':   0.0,
                        'cp':    0.0,
                        'k':     0.0,
                    }
                except (ValueError, TypeError) as e:
                    print(f"[ic line {line_nr + 1}] defmat({mat_name}): {e}")
            elif len(resolved) == 6:
                # sigma, alpha, mi_r, rho, cp, thermal_k
                try:
                    custom_materials[mat_name] = {
                        'sigma': float(resolved[0]),
                        'alpha': float(resolved[1]),
                        'mi_r':  float(resolved[2]),
                        'rho':   float(resolved[3]),
                        'cp':    float(resolved[4]),
                        'k':     float(resolved[5]),
                    }
                except (ValueError, TypeError) as e:
                    print(f"[ic line {line_nr + 1}] defmat({mat_name}): {e}")
            else:
                print(f"[ic line {line_nr + 1}] defmat({mat_name}): expected 3 or 6 numeric args, got {len(resolved)}")

        elif command in _ANALYSIS_KEYS:
            # freq(v) | length(v) | temp(v) | htc(v) | cellsize(v)
            try:
                analysis_params[command] = _eval(raw_args.strip(), innerVariables)
            except Exception as e:
                print(f"[ic line {line_nr + 1}] Error parsing {command}({raw_args}): {e}")

        else:
            # Geometry commands: c, r, mv, cp
            ar = [a.strip() for a in raw_args.split(',')]
            resolved = _resolve_args(ar, innerVariables)
            if len(resolved) in commands[command][1]:
                try:
                    float_args = [float(a) for a in resolved]
                    innerCodeSteps.append([commands[command][0], float_args, command])
                except (ValueError, TypeError) as e:
                    print(f"[ic line {line_nr + 1}] Cannot convert args to numbers "
                          f"in '{line}': {e}")

    return innerCodeSteps, currents, materials, custom_materials, analysis_params
