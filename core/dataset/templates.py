"""
Procedural TikZ template generators for synthetic dataset construction.

Every generator is a pure function of a seeded numpy Generator: no I/O, no
global mutable state, deterministic under a fixed seed. All geometry lives
inside a [-5, 5]^2 canvas with 2-decimal coordinate precision so that every
sample stays well under the 512-token encoder budget (proxy: 4000 chars).

Reference: Golub & Van Loan, Matrix Computations — vectorized sampling of
vertex coordinates via trigonometric evaluation over batched angle arrays.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.exceptions import DomainError

NAMED_COLORS: tuple[str, ...] = (
    "red",
    "blue",
    "green",
    "black",
    "orange",
    "purple",
    "brown",
    "cyan",
    "magenta",
    "gray",
)
LINE_STYLES: tuple[str, ...] = ("solid", "dashed", "dotted")
LINE_WIDTHS: tuple[str, ...] = ("thin", "thick", "very thick")
NODE_LABELS: tuple[str, ...] = ("A", "B", "C", "P", "Q", "x", "y", "z")

FAMILY_NAMES: tuple[str, ...] = (
    "line_segment",
    "polyline",
    "polygon",
    "circle_arc",
    "grid_axes",
    "function_plot",
    "node_arrow",
    "composed",
)

CANVAS_BOUND: float = 5.0


def _format_scalar(value: float) -> str:
    """
    Formats a coordinate with fixed 2-decimal precision.

    Temporal complexity: O(1).
    """
    return f"{value + 0.0:.2f}"


def _format_point(x: float, y: float) -> str:
    """
    Formats a 2D TikZ coordinate literal.

    Temporal complexity: O(1).
    """
    return f"({_format_scalar(x)}, {_format_scalar(y)})"


def _sample_points(rng: np.random.Generator, count: int) -> NDArray[Any]:
    """
    Samples `count` uniform points in the [-5, 5]^2 canvas.

    Returns:
        np.ndarray: Point array. Shape: (count, 2)

    Temporal complexity: O(count), vectorized in a single RNG draw.
    """
    # Shape: (count, 2)
    return rng.uniform(-CANVAS_BOUND, CANVAS_BOUND, size=(count, 2))


def _choice(rng: np.random.Generator, options: tuple[str, ...]) -> str:
    """
    Draws one element uniformly from a finite option tuple.

    Temporal complexity: O(1).
    """
    return options[int(rng.integers(0, len(options)))]


def _point_chain(points: NDArray[Any]) -> str:
    """
    Serializes a point array into a TikZ `--` connected path body.

    Temporal complexity: O(k) where k is the number of points (k <= 12).
    """
    return " -- ".join(_format_point(float(point[0]), float(point[1])) for point in points)


def _wrap_picture(*commands: str) -> str:
    """
    Wraps drawing commands into a complete tikzpicture environment.

    Temporal complexity: O(k) where k is the number of commands.
    """
    body: str = "\n".join(commands)
    return f"\\begin{{tikzpicture}}\n{body}\n\\end{{tikzpicture}}"


def _line_segment(rng: np.random.Generator) -> str:
    """
    Generates a single styled line segment.

    Temporal complexity: O(1).
    """
    points: NDArray[Any] = _sample_points(rng, 2)
    command: str = (
        f"\\draw[{_choice(rng, LINE_WIDTHS)}, {_choice(rng, NAMED_COLORS)}, "
        f"{_choice(rng, LINE_STYLES)}] {_point_chain(points)};"
    )
    return _wrap_picture(command)


def _polyline(rng: np.random.Generator) -> str:
    """
    Generates an open polyline of 3 to 6 vertices.

    Temporal complexity: O(k) where k is the vertex count.
    """
    vertex_count: int = int(rng.integers(3, 7))
    points: NDArray[Any] = _sample_points(rng, vertex_count)
    command: str = (
        f"\\draw[{_choice(rng, LINE_WIDTHS)}, {_choice(rng, NAMED_COLORS)}] {_point_chain(points)};"
    )
    return _wrap_picture(command)


def _polygon(rng: np.random.Generator) -> str:
    """
    Generates a regular polygon of 3 to 8 sides via vectorized vertex evaluation.

    Vertices are (cx + r*cos(theta_k), cy + r*sin(theta_k)) over a batched
    angle array; no scalar iteration over vertices is performed.

    Temporal complexity: O(n) where n is the side count, vectorized.
    """
    sides: int = int(rng.integers(3, 9))
    center: NDArray[Any] = _sample_points(rng, 1)[0]
    radius: float = float(rng.uniform(0.5, 3.0))
    rotation: float = float(rng.uniform(0.0, 2.0 * np.pi))

    # Shape: (sides,)
    angles: NDArray[Any] = rotation + 2.0 * np.pi * np.arange(sides) / sides
    # Shape: (sides, 2)
    vertices: NDArray[Any] = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))

    command: str = (
        f"\\draw[{_choice(rng, LINE_WIDTHS)}, {_choice(rng, NAMED_COLORS)}] "
        f"{_point_chain(vertices)} -- cycle;"
    )
    return _wrap_picture(command)


def _circle_arc(rng: np.random.Generator) -> str:
    """
    Generates a circle, ellipse, or angular arc.

    Temporal complexity: O(1).
    """
    center: NDArray[Any] = _sample_points(rng, 1)[0]
    style: str = f"{_choice(rng, LINE_WIDTHS)}, {_choice(rng, NAMED_COLORS)}"
    variant: int = int(rng.integers(0, 3))

    if variant == 0:
        radius: float = float(rng.uniform(0.5, 3.5))
        command = (
            f"\\draw[{style}] {_format_point(float(center[0]), float(center[1]))} "
            f"circle ({_format_scalar(radius)});"
        )
    elif variant == 1:
        semi_a: float = float(rng.uniform(0.5, 3.5))
        semi_b: float = float(rng.uniform(0.5, 3.5))
        command = (
            f"\\draw[{style}] {_format_point(float(center[0]), float(center[1]))} "
            f"ellipse ({_format_scalar(semi_a)} and {_format_scalar(semi_b)});"
        )
    else:
        start_angle: float = float(rng.uniform(0.0, 360.0))
        end_angle: float = start_angle + float(rng.uniform(30.0, 330.0))
        radius = float(rng.uniform(0.5, 3.5))
        command = (
            f"\\draw[{style}] {_format_point(float(center[0]), float(center[1]))} "
            f"arc ({_format_scalar(start_angle)}:{_format_scalar(end_angle)}:"
            f"{_format_scalar(radius)});"
        )

    return _wrap_picture(command)


def _grid_axes(rng: np.random.Generator) -> str:
    """
    Generates a stepped coordinate grid with arrowed Cartesian axes.

    Temporal complexity: O(1).
    """
    step: float = float(rng.uniform(0.5, 2.0))
    bound: float = float(rng.uniform(3.0, CANVAS_BOUND))
    corner_lo: str = _format_point(-bound, -bound)
    corner_hi: str = _format_point(bound, bound)
    axis_h: str = (
        f"\\draw[->, thick] {_format_point(-CANVAS_BOUND, 0.0)} -- "
        f"{_format_point(CANVAS_BOUND, 0.0)};"
    )
    axis_v: str = (
        f"\\draw[->, thick] {_format_point(0.0, -CANVAS_BOUND)} -- "
        f"{_format_point(0.0, CANVAS_BOUND)};"
    )
    grid: str = f"\\draw[step={_format_scalar(step)}, gray, thin] {corner_lo} grid {corner_hi};"
    return _wrap_picture(grid, axis_h, axis_v)


def _function_plot(rng: np.random.Generator) -> str:
    """
    Generates a smooth function plot over a symmetric domain.

    Temporal complexity: O(1).
    """
    amplitude: float = float(rng.uniform(0.5, 3.0))
    frequency: float = float(rng.uniform(0.5, 3.0))
    bound: float = float(rng.uniform(3.0, CANVAS_BOUND))

    basis: str = _choice(
        rng,
        (
            f"{_format_scalar(amplitude)}*sin({_format_scalar(frequency)}*\\x r)",
            f"{_format_scalar(amplitude)}*cos({_format_scalar(frequency)}*\\x r)",
            f"{_format_scalar(0.2 * amplitude)}*(\\x)^2",
            f"{_format_scalar(amplitude)}*\\x",
        ),
    )
    command: str = (
        f"\\draw[domain={_format_scalar(-bound)}:{_format_scalar(bound)}, smooth, "
        f"{_choice(rng, LINE_WIDTHS)}, {_choice(rng, NAMED_COLORS)}] "
        f"plot (\\x, {{{basis}}});"
    )
    return _wrap_picture(command)


def _node_arrow(rng: np.random.Generator) -> str:
    """
    Generates a labeled node diagram connected by directed edges.

    Temporal complexity: O(k) where k is the node count (k <= 4).
    """
    node_count: int = int(rng.integers(2, 5))
    points: NDArray[Any] = _sample_points(rng, node_count)
    identifiers: tuple[str, ...] = tuple("abcde"[:node_count])

    declarations: list[str] = [
        f"\\node[circle, draw=black] ({identifiers[idx]}) at "
        f"{_format_point(float(points[idx][0]), float(points[idx][1]))} "
        f"{{{_choice(rng, NODE_LABELS)}}};"
        for idx in range(node_count)
    ]
    edges: list[str] = [
        f"\\draw[->, thick, {_choice(rng, NAMED_COLORS)}] "
        f"({identifiers[idx]}) -- ({identifiers[idx + 1]});"
        for idx in range(node_count - 1)
    ]
    return _wrap_picture(*(declarations + edges))


def _extract_body(markup: str) -> str:
    """
    Extracts the command body from a wrapped tikzpicture markup.

    Temporal complexity: O(L) where L is the markup length.
    """
    return markup.split("\n", 1)[1].rsplit("\n", 1)[0]


def _composed(rng: np.random.Generator) -> str:
    """
    Generates a composition of two to three disjoint primitives.

    Temporal complexity: O(1) per primitive.
    """
    primitive_pool: tuple[Callable[[np.random.Generator], str], ...] = (
        _line_segment,
        _polygon,
        _circle_arc,
    )
    primitive_count: int = int(rng.integers(2, 4))
    # Shape: (primitive_count,)
    selected: NDArray[Any] = rng.choice(len(primitive_pool), size=primitive_count)

    commands: list[str] = [
        _extract_body(primitive_pool[int(primitive_idx)](rng)) for primitive_idx in selected
    ]
    return _wrap_picture(*commands)


_GENERATORS: dict[str, Callable[[np.random.Generator], str]] = {
    "line_segment": _line_segment,
    "polyline": _polyline,
    "polygon": _polygon,
    "circle_arc": _circle_arc,
    "grid_axes": _grid_axes,
    "function_plot": _function_plot,
    "node_arrow": _node_arrow,
    "composed": _composed,
}


def generate_sample(family: str, rng: np.random.Generator) -> str:
    """
    Generates one procedural TikZ sample from the requested family.

    Args:
        family (str): Family identifier, must be listed in FAMILY_NAMES.
        rng (np.random.Generator): Seeded random generator (pure input).

    Returns:
        str: Complete tikzpicture markup.

    Raises:
        DomainError: If the family identifier is unknown.

    Temporal complexity: O(1) amortized per sample.
    """
    generator: Callable[[np.random.Generator], str] | None = _GENERATORS.get(family)
    if generator is None:
        raise DomainError(
            f"Unknown template family '{family}'. Valid families: {list(FAMILY_NAMES)}."
        )
    return generator(rng)


def generate_batch(family: str, count: int, seed: int) -> list[str]:
    """
    Generates `count` deterministic samples from one family.

    Args:
        family (str): Family identifier, must be listed in FAMILY_NAMES.
        count (int): Number of samples to draw. Must be positive.
        seed (int): Seed for the local Generator; fixes the entire batch.

    Returns:
        list[str]: Deterministic sequence of tikzpicture markups.

    Raises:
        DomainError: If the family is unknown or count is non-positive.

    Temporal complexity: O(count) amortized.
    """
    if count <= 0:
        raise DomainError(f"Sample count must be positive. Got {count}.")

    rng: np.random.Generator = np.random.default_rng(seed)
    return [generate_sample(family, rng) for _ in range(count)]


def family_index(family: str) -> int:
    """
    Maps a family identifier to its stratification index.

    Args:
        family (str): Family identifier, must be listed in FAMILY_NAMES.

    Returns:
        int: Zero-based family index.

    Raises:
        DomainError: If the family identifier is unknown.

    Temporal complexity: O(F) where F is the family count (F = 8).
    """
    if family not in FAMILY_NAMES:
        raise DomainError(
            f"Unknown template family '{family}'. Valid families: {list(FAMILY_NAMES)}."
        )
    return FAMILY_NAMES.index(family)
