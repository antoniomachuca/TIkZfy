"""
Compositional (SCFG/AST) TikZ generator for Tier 2 synthetic data.

A weighted stochastic context-free grammar expands a ``Figure`` non-terminal
into a variable number of drawing scopes. Unlike the linear templates in
``templates.py``, every production is a probabilistic choice so two figures
rarely share an AST, yet a fixed seed reproduces byte-identical markup.

Grammar (weighted productions):

    Figure    -> Scope | Figure Scope
    Scope     -> DrawCmd | NodeCmd | FillCmd | PlotCmd
    DrawCmd   -> ('\\draw' | '\\path[draw') StyleList PathExpr ';'
    PathExpr  -> Coord (Connector Coord)* ['arc' ArcSpec]
    Coord     -> AbsCoord | RelCoord
    AbsCoord  -> '(' float ',' float ')'
    RelCoord  -> '++(' float ',' float ')'
    StyleList -> '[' Style (',' Style)* ']'
    Style     -> Color | LineWidth | LineStyle | Arrow | Fill | RoundedCorners
    NodeCmd   -> '\\node' StyleList 'at' Coord '{' Label '}' ';'
    FillCmd   -> '\\fill' StyleList PathExpr '--cycle' ';'
    PlotCmd   -> '\\draw' '[domain=..., smooth, ...] plot (\\x, {expr})' ';'

References:
    Chomsky, Three Models for the Description of Language — context-free
        grammars as the generative backbone of the synthetic corpus.
    Golub & Van Loan, Matrix Computations — vectorized coordinate sampling
        over batched uniform draws from the seeded ``np.random.Generator``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.dataset.curation import within_length_budget
from core.dataset.templates import LINE_STYLES, LINE_WIDTHS, NAMED_COLORS
from core.exceptions import DomainError

CANVAS_BOUND: float = 5.0

DEFAULT_DEPTH_RANGE: tuple[int, int] = (5, 20)

# Weighted arrow-tip production. The empty string (no arrow) dominates so a
# majority of paths remain undirected line work.
_ARROW_OPTIONS: tuple[str, ...] = ("", "->", "->>", "<->", "-stealth")
_ARROW_WEIGHTS: tuple[float, ...] = (0.5, 0.2, 0.1, 0.1, 0.1)

# Weighted orthogonal/continuous connectors. ``--`` dominates; ``|-`` and
# ``-|`` introduce right-angle polyline segments.
_CONNECTOR_OPTIONS: tuple[str, ...] = ("--", "|-", "-|")
_CONNECTOR_WEIGHTS: tuple[float, ...] = (0.8, 0.1, 0.1)

# Weighted node shapes for the ``\\node`` production.
_NODE_SHAPE_OPTIONS: tuple[str, ...] = ("circle", "rectangle", "diamond", "ellipse")
_NODE_SHAPE_WEIGHTS: tuple[float, ...] = (0.4, 0.3, 0.15, 0.15)

_NODE_LABELS: tuple[str, ...] = ("A", "B", "C", "P", "Q", "x", "y", "z", "m", "n")

# Weighted fill patterns. ``none`` (plain solid fill) is most common; the
# remaining entries require the ``patterns`` TikZ library.
_PATTERN_OPTIONS: tuple[str, ...] = ("none", "north east lines", "crosshatch", "dots")
_PATTERN_WEIGHTS: tuple[float, ...] = (0.7, 0.15, 0.1, 0.05)

# Weighted plot basis expressions, mirroring ``templates._function_plot``.
_PLOT_BASIS_OPTIONS: tuple[str, ...] = (
    "sin", "cos", "quadratic", "linear",
)
_PLOT_BASIS_WEIGHTS: tuple[float, ...] = (0.3, 0.3, 0.2, 0.2)


def _format_scalar(value: float) -> str:
    """Formats a coordinate with fixed 2-decimal precision. O(1)."""
    return f"{value + 0.0:.2f}"


def _format_point(x: float, y: float) -> str:
    """Formats a 2D TikZ coordinate literal. O(1)."""
    return f"({_format_scalar(x)}, {_format_scalar(y)})"


def _sample_points(rng: np.random.Generator, count: int) -> NDArray[Any]:
    """Samples ``count`` uniform points in the [-5, 5]^2 canvas. O(count)."""
    return rng.uniform(-CANVAS_BOUND, CANVAS_BOUND, size=(count, 2))


def _uniform_choice(rng: np.random.Generator, options: tuple[str, ...]) -> str:
    """Draws one element uniformly from a finite option tuple. O(1)."""
    return options[int(rng.integers(0, len(options)))]


def _weighted_choice(
    rng: np.random.Generator, options: tuple[str, ...], weights: tuple[float, ...]
) -> str:
    """Draws one element from ``options`` with normalized ``weights``. O(k)."""
    probabilities: NDArray[Any] = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return options[int(rng.choice(len(options), p=probabilities))]


def _color(rng: np.random.Generator) -> str:
    """Draws a named color uniformly from the shared color palette. O(1)."""
    return _uniform_choice(rng, NAMED_COLORS)


def _coord(rng: np.random.Generator) -> str:
    """Emits an absolute or relative coordinate literal. O(1)."""
    x: float = float(rng.uniform(-CANVAS_BOUND, CANVAS_BOUND))
    y: float = float(rng.uniform(-CANVAS_BOUND, CANVAS_BOUND))
    if rng.random() < 0.5:
        return _format_point(x, y)
    return f"++{_format_point(x, y)}"


def _connector(rng: np.random.Generator) -> str:
    """Draws a weighted path connector (``--`` dominant). O(1)."""
    return _weighted_choice(rng, _CONNECTOR_OPTIONS, _CONNECTOR_WEIGHTS)


def _arc_spec(rng: np.random.Generator) -> str:
    """Emits an ``arc`` segment specification. O(1)."""
    start: float = float(rng.uniform(0.0, 360.0))
    end: float = start + float(rng.uniform(30.0, 330.0))
    radius: float = float(rng.uniform(0.5, 3.5))
    return (
        f"arc ({_format_scalar(start)}:{_format_scalar(end)}:"
        f"{_format_scalar(radius)})"
    )


def _path_expr(rng: np.random.Generator, close: bool = False) -> str:
    """Emits a chained coordinate path, optionally closed with ``-- cycle``. O(k)."""
    segment_count: int = int(rng.integers(2, 7))
    coordinates: list[str] = [_coord(rng) for _ in range(segment_count)]
    connectors: list[str] = [_connector(rng) for _ in range(segment_count - 1)]

    body: str = coordinates[0]
    for coordinate, connector in zip(coordinates[1:], connectors, strict=True):
        body = f"{body} {connector} {coordinate}"
    if rng.random() < 0.25:
        body = f"{body} {_arc_spec(rng)}"
    if close:
        body = f"{body} -- cycle"
    return body


def _draw_style_list(rng: np.random.Generator) -> list[str]:
    """Builds a weighted style token list for a draw/plot command. O(1)."""
    styles: list[str] = [_color(rng)]
    if rng.random() < 0.7:
        styles.append(_uniform_choice(rng, LINE_WIDTHS))
    if rng.random() < 0.5:
        styles.append(_uniform_choice(rng, LINE_STYLES))
    if rng.random() < 0.4:
        styles.append(_weighted_choice(rng, _ARROW_OPTIONS, _ARROW_WEIGHTS))
    if rng.random() < 0.2:
        styles.append("rounded corners")
    return [style for style in styles if style]


def _draw_cmd(rng: np.random.Generator) -> str:
    """Emits a ``\\draw`` or syntactically-equivalent ``\\path[draw]``. O(k)."""
    styles: list[str] = _draw_style_list(rng)
    path: str = _path_expr(rng)
    if rng.random() < 0.2:
        return f"\\path[draw, {', '.join(styles)}] {path};"
    return f"\\draw[{', '.join(styles)}] {path};"


def _fill_cmd(rng: np.random.Generator) -> str:
    """Emits a ``\\fill`` over a closed path with opacity/pattern. O(k)."""
    color: str = _color(rng)
    opacity: float = float(rng.uniform(0.2, 0.9))
    pattern: str = _weighted_choice(rng, _PATTERN_OPTIONS, _PATTERN_WEIGHTS)
    styles: list[str] = [color, f"fill opacity={_format_scalar(opacity)}"]
    if pattern != "none":
        styles.append(f"pattern={pattern}")
    path: str = _path_expr(rng, close=True)
    return f"\\fill[{', '.join(styles)}] {path};"


def _node_cmd(rng: np.random.Generator) -> str:
    """Emits a ``\\node`` with a shape, border, and fill. O(1)."""
    shape: str = _weighted_choice(rng, _NODE_SHAPE_OPTIONS, _NODE_SHAPE_WEIGHTS)
    color: str = _color(rng)
    label: str = _uniform_choice(rng, _NODE_LABELS)
    styles: list[str] = [shape, f"draw={color}"]
    if rng.random() < 0.6:
        styles.append(f"fill={_color(rng)}")
    if rng.random() < 0.4:
        styles.append(_uniform_choice(rng, LINE_WIDTHS))
    return f"\\node[{', '.join(styles)}] at {_coord(rng)} {{{label}}};"


def _plot_cmd(rng: np.random.Generator) -> str:
    """Emits a ``plot`` of a weighted analytic basis. O(1)."""
    amplitude: float = float(rng.uniform(0.5, 3.0))
    frequency: float = float(rng.uniform(0.5, 3.0))
    bound: float = float(rng.uniform(3.0, CANVAS_BOUND))
    basis: str = _weighted_choice(rng, _PLOT_BASIS_OPTIONS, _PLOT_BASIS_WEIGHTS)
    color: str = _color(rng)
    expression: str = {
        "sin": f"{_format_scalar(amplitude)}*sin({_format_scalar(frequency)}*\\x r)",
        "cos": f"{_format_scalar(amplitude)}*cos({_format_scalar(frequency)}*\\x r)",
        "quadratic": f"{_format_scalar(0.2 * amplitude)}*(\\x)^2",
        "linear": f"{_format_scalar(amplitude)}*\\x",
    }[basis]
    return (
        f"\\draw[domain={_format_scalar(-bound)}:{_format_scalar(bound)}, smooth, "
        f"{color}] plot (\\x, {{{expression}}});"
    )


_SCOPE_OPTIONS: tuple[str, ...] = ("draw", "node", "fill", "plot")
_SCOPE_WEIGHTS: tuple[float, ...] = (0.5, 0.2, 0.2, 0.1)

_SCOPE_GENERATORS: dict[str, Callable[[np.random.Generator], str]] = {
    "draw": _draw_cmd,
    "node": _node_cmd,
    "fill": _fill_cmd,
    "plot": _plot_cmd,
}


def _scope(rng: np.random.Generator) -> str:
    """Expands one ``Scope`` non-terminal into a weighted drawing command. O(k)."""
    kind: str = _weighted_choice(rng, _SCOPE_OPTIONS, _SCOPE_WEIGHTS)
    return _SCOPE_GENERATORS[kind](rng)


def _wrap_picture(commands: list[str]) -> str:
    """Wraps a list of drawing commands into a tikzpicture environment. O(k)."""
    body: str = "\n".join(commands)
    return f"\\begin{{tikzpicture}}\n{body}\n\\end{{tikzpicture}}"


def generate_compositional_sample(
    rng: np.random.Generator, depth_range: tuple[int, int] = DEFAULT_DEPTH_RANGE
) -> str:
    """
    Generates one compositional TikZ figure from the weighted SCFG.

    Args:
        rng (np.random.Generator): Seeded random generator (pure input).
        depth_range (tuple[int, int]): Inclusive ``(min, max)`` count of drawing
            scopes (primitives) per figure.

    Returns:
        str: Complete tikzpicture markup guaranteed to fit the decoder budget.

    Raises:
        DomainError: On an invalid depth range or a figure that exceeds the
            length budget (defensive guard; should not occur for sane ranges).

    Temporal complexity: O(D) where D is the sampled scope count.
    """
    minimum, maximum = depth_range
    if minimum < 1 or maximum < minimum:
        raise DomainError(
            f"Invalid depth_range {depth_range}; expected 1 <= min <= max."
        )

    scope_count: int = int(rng.integers(minimum, maximum + 1))
    commands: list[str] = [_scope(rng) for _ in range(scope_count)]
    markup: str = _wrap_picture(commands)

    if not within_length_budget(markup):
        raise DomainError(
            "Generated compositional markup exceeded the decoder length budget."
        )
    return markup


def generate_compositional_batch(count: int, seed: int) -> list[str]:
    """
    Generates ``count`` deterministic compositional figures from one seed.

    Args:
        count (int): Number of samples to draw. Must be positive.
        seed (int): Seed for the local Generator; fixes the entire batch.

    Returns:
        list[str]: Deterministic sequence of tikzpicture markups.

    Raises:
        DomainError: If ``count`` is non-positive.

    Temporal complexity: O(count) amortized.
    """
    if count <= 0:
        raise DomainError(f"Sample count must be positive. Got {count}.")

    rng: np.random.Generator = np.random.default_rng(seed)
    return [generate_compositional_sample(rng) for _ in range(count)]
