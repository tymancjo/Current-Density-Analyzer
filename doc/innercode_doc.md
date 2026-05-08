# Inner-Code (`.ic`) File Reference

`.ic` files are the case-description language used by the Current Density Analyzer.
Each file fully describes a cross-section: its geometry (conductors, structural parts),
the material of every conductor, and the electrical excitation (current, phase).

---

## File structure

An `.ic` file is plain text. Lines are processed in order. There are four kinds of content:

| Kind | Syntax | Purpose |
|---|---|---|
| Comment | `# any text` | Documentation; ignored by the parser |
| Variable | `v(…)` / `a(…)` | Define and update numeric variables |
| Physical setup | `current(…)` / `material(…)` | Assign electrical properties to phases |
| Geometry | `r(…)` / `c(…)` / `mv(…)` / `cp(…)` | Draw shapes on the cross-section canvas |

There is no required ordering — `current` and `material` lines may appear anywhere,
including inside loops. The parser collects them all before the solver runs.

---

## Comments

```
# This line is ignored
r(0, 0, 10, 60, 1)   # inline comments are NOT supported — put # on its own line
```

---

## Variables

### `v(name, expression)` — declare / set

Creates (or overwrites) a variable called `name` and sets it to the result of `expression`.

```
v(x, 0)          # x = 0
v(w, 10)         # w = 10
v(h, 60)         # h = 60
v(area, w*h)     # area = 600  — expression referencing earlier variables
v(cx, x + w/2)   # cx = 5.0
v(r, sqrt(area)) # r ≈ 24.49  — math functions available
```

### `a(name, expression)` — accumulate (add)

Adds the result of `expression` to an already-declared variable.
`a` is short for "add" — think of it as `name += expression`.

```
v(x, 0)
a(x, 20)    # x is now 20
a(x, 20)    # x is now 40
a(x, -5)    # x is now 35   — subtraction via negative delta
a(x, w*2)   # x += 2*w      — expressions allowed
```

`a()` requires the variable to be declared with `v()` first.
Using `a()` on an undefined variable prints a warning and skips the line.

### Available math in expressions

All standard arithmetic operators work: `+  -  *  /  **  %  //`

The following functions and constants are available by name:

| Function / Constant | Description |
|---|---|
| `sin(x)`, `cos(x)`, `tan(x)` | Trigonometry (radians) |
| `asin(x)`, `acos(x)`, `atan(x)`, `atan2(y,x)` | Inverse trig |
| `sqrt(x)` | Square root |
| `exp(x)`, `log(x)`, `log10(x)` | Exponential / logarithm |
| `abs(x)` | Absolute value |
| `round(x)`, `ceil(x)`, `floor(x)` | Rounding |
| `min(a,b)`, `max(a,b)` | Minimum / maximum |
| `int(x)`, `float(x)` | Type conversion |
| `pi` | 3.14159… |
| `e` | 2.71828… |
| `math.sin(x)` etc. | Full `math` module also available |

> **Tip — degrees to radians:** `sin(angle * pi / 180)` or `sin(math.radians(angle))`

---

## Physical setup

### `current(phase_id, I_rms, phase_shift, extra_shift)`

Assigns electrical excitation to a phase.

| Argument | Unit | Meaning |
|---|---|---|
| `phase_id` | integer | Must match an ID used in geometry commands |
| `I_rms` | A (ampere) | RMS current magnitude |
| `phase_shift` | degrees | Phase angle (e.g. 0, 120, 240 for balanced 3-phase) |
| `extra_shift` | degrees | Additional shift; use 180 for a return conductor |

```
current(1, 2300, 0,    0)    # phase 1: 2300 A ∠0°
current(2, 2300, 120,  0)    # phase 2: 2300 A ∠120°
current(3, 2300, 240,  0)    # phase 3: 2300 A ∠240°
current(4, 2300, 0,  180)    # return of phase 1
current(99, 0.0001, 0, 180)  # structural part — effectively zero current
```

### `material(phase_id, material_type_id)`

Assigns a material to a phase. The `material_type_id` is the row index (0-based)
in `setup/materials.txt`:

| ID | Name | σ [S/m] | Typical use |
|---|---|---|---|
| 0 | Copper | 56×10⁶ | Busbars, cables |
| 1 | FakeAl | 31×10⁶ | Placeholder aluminium |
| 2 | Aluminium | 31×10⁶ | Al conductors |
| 3 | Carbon Steel | 6.99×10⁶ | Structural steel |
| 4 | Stainless Steel | 1.45×10⁶ | Non-magnetic steel |
| 5 | Fake Carbon Steel | 6.99×10⁶ | Short-section approximation |

```
material(1, 0)   # phase 1 is copper
material(2, 2)   # phase 2 is aluminium
material(99, 3)  # structural frame is carbon steel
```

---

## Geometry commands

All dimensions are in **millimetres**. The origin (0, 0) is the bottom-left corner.
The canvas is automatically sized to fit all drawn shapes.

### `r(x, y, width, height, phase_id)` — Rectangle

Draws a filled rectangle.
`(x, y)` is the **bottom-left corner**; `width` and `height` extend in the positive direction.

```
r(0,   0,  10, 60, 1)   # 10 mm × 60 mm rectangle, phase 1, at origin
r(x,   y,   w,  h, p)   # same using variables
r(x+5, y,   w,  h, p)   # x-position offset by expression
```

### `c(x, y, D, phase_id)` — Solid circle

Draws a filled disc centred at `(x, y)` with diameter `D`.

```
c(100, 50, 20.5, 1)    # Ø20.5 mm disc at (100, 50), phase 1
c(cx, cy, fi, phA)     # using variables
```

### `c(x, y, D_outer, id_outer, D_inner, id_inner)` — Annulus (hollow circle)

Draws a ring: the annulus between `D_inner` and `D_outer` gets `id_outer`;
the inner disc gets `id_inner`.

```
c(447, 0, 75, 9, 70, 0)   # outer ring: Ø75 mm phase 9; inner core: Ø70 mm air (0)
```

> Set `id_inner = 0` to punch a hole through a previously drawn shape.

### `mv(phase_id, shift_x, shift_y)` — Move

Shifts all cells belonging to `phase_id` by `(shift_x, shift_y)` mm.
The original cells are removed.

```
mv(1, 20, 0)    # shift phase 1 by 20 mm to the right
```

### `cp(phase_id, shift_x, shift_y)` — Copy

Same as `mv` but the original cells remain in place.

```
cp(1, 20, 0)    # duplicate phase 1, copy sits 20 mm to the right
```

---

## Loops

### `l(n)` … `break`

Repeats the enclosed block **n times** in total.

```
v(x, 0)
l(3)
    r(x, 0, 10, 60, 1)
    a(x, 20)
break
```

This draws three rectangles at x = 0, 20, 40.

**Rules:**
- `l(n)` must be matched by a `break`.
- The body executes exactly `n` times (including the first execution).
- `l(1)` runs the body once (same as no loop).
- Variable changes inside the loop persist into the next iteration and beyond.

### Nested loops

```
v(x,  0)
v(y,  0)
v(p,  1)

l(3)               # 3 rows
    v(x, 0)        # reset x for every row
    l(4)           # 4 columns
        r(x, y, 10, 60, p)
        a(x, 20)
    break          # end inner loop
    a(y, 100)
    a(p, 1)
break              # end outer loop
```

> Each `break` terminates the innermost active `l()` block.

### Phase variables inside loops

Variables can hold phase IDs, making it easy to cycle through many phases:

```
v(ph, 1)
v(x,  0)

l(6)
    current(ph, 1000, 0, 0)
    material(ph, 0)
    r(x, 0, 10, 60, ph)
    a(ph, 1)
    a(x,  20)
break
```

---

## Phase ID conventions

Phase IDs are integers you choose freely. Conventional ranges:

| Range | Typical use |
|---|---|
| 1 – 3 | Three-phase conductors (L1, L2, L3) |
| 1 – 6 | Two three-phase systems or 6-conductor arrangements |
| 11 – 13 | Second busbar system |
| 21 – 23 | Third system or auxiliary cables |
| 98, 99 | Structural / frame elements (assign near-zero current) |

The phase IDs used in `current()`, `material()`, and geometry commands must all match.

---

## Complete examples

### Example 1 — Simple three-phase flat busbar

```
# Three aluminium busbars, 10 × 60 mm, 20 mm pitch

material(1, 2)
material(2, 2)
material(3, 2)

current(1, 2300, 0,   0)
current(2, 2300, 120, 0)
current(3, 2300, 240, 0)

v(p, 1)
v(x, 0)

l(3)
    r(x, 0, 10, 60, p)
    a(x, 20)
    a(p, 1)
break
```

### Example 2 — Double busbar system in a steel frame

```
# Two vertically stacked busbars per phase, steel enclosure

material(1, 2)   material(2, 2)   material(3, 2)
material(11, 2)  material(12, 2)  material(13, 2)
material(98, 3)  material(99, 3)

current(1,  2300, -120, 0)  current(2,  2300, 0, 0)  current(3,  2300, 120, 0)
current(11, 2700, -120, 0)  current(12, 2700, 0, 0)  current(13, 2700, 120, 0)
current(98, 0.0001, 0, 180)
current(99, 0.0001, 0, 180)

# Steel side frames
r(-40, -40, 5, 610, 98)
r(240, -40, 5, 610, 99)

v(p0, 1)   v(p1, 11)
v(x, 0)    v(y1, 0)   v(y2, 80)
v(w, 10)   v(h, 60)

l(3)
    r(x,    y1, w, h, p0)
    r(x+20, y1, w, h, p0)
    r(x,    y2, w, h, p1)
    r(x+20, y2, w, h, p1)
    a(x,  150)
    a(p0, 1)
    a(p1, 1)
    v(x,  0)
    a(y1, 200)
    a(y2, 200)
break
```

### Example 3 — Cables with expressions and trig

```
# Five cable modules, three cables per module arranged in an equilateral triangle

v(I,    238)
v(D,    20.5)
v(step, 60)      # vertical pitch between modules [mm]
v(r,    30)      # triangle radius [mm]

v(ph,  1)
v(y,   0)

l(5)
    material(ph,   0)
    material(ph+1, 0)
    material(ph+2, 0)

    current(ph,   I, -120, 0)
    current(ph+1, I,    0, 0)
    current(ph+2, I,  120, 0)

    # equilateral triangle positions
    c(r*cos(270*pi/180),        y + r*sin(270*pi/180),        D, ph)
    c(r*cos(270*pi/180 + 2*pi/3), y + r*sin(270*pi/180 + 2*pi/3), D, ph+1)
    c(r*cos(270*pi/180 + 4*pi/3), y + r*sin(270*pi/180 + 4*pi/3), D, ph+2)

    a(ph, 3)
    a(y,  step)
break
```

---

## Tips and common pitfalls

**Variables must be declared before use**
`a(x, 10)` before `v(x, 0)` will print a warning and be skipped.

**`a()` is always relative, `v()` is always absolute**
To reset `x` inside a loop use `v(x, 0)`, not `a(x, ...)`.

**Phase IDs are just numbers — make sure they are consistent**
A `current()` with a phase ID that has no matching geometry cell will be silently ignored by the solver.

**Expressions with nested commas in multi-argument positions are not supported**
`r(max(a,b), y, w, h, p)` will not parse correctly because the comma inside `max()` confuses the argument splitter.
Use a variable instead: `v(xpos, max(a,b))` then `r(xpos, y, w, h, p)`.

**Expressions in `v()` may freely contain commas**
`v(xpos, max(a, b))` works because the value is always the second argument and everything after the first comma is treated as one expression.

**The loop body runs `n` times in total**
`l(1)` runs once. `l(0)` also runs once (edge case — avoid it). Use `l(n)` where n ≥ 2 for actual repetition.

**There is no conditional (`if`)**
Work around this with careful variable initialisation and the fact that drawing phase 0 (background) effectively erases a cell.
