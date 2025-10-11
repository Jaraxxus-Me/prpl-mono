"""Tests for tomsgeoms2d."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tomsgeoms2d.structs import (
    Circle,
    LineSegment,
    Lobject,
    Rectangle,
    RTrapezoid,
    Triangle,
)
from tomsgeoms2d.utils import geom2ds_intersect


def test_line_segment():
    """Tests for LineSegment()."""
    _, ax = plt.subplots(1, 1)
    ax.set_xlim((-5, 5))
    ax.set_ylim((-8, 8))

    seg1 = LineSegment(x1=0, y1=1, x2=3, y2=7)
    assert seg1.x1 == 0
    assert seg1.y1 == 1
    assert seg1.x2 == 3
    assert seg1.y2 == 7
    seg1.plot(ax, color="red", linewidth=2)
    assert seg1.contains_point(2, 5)
    assert not seg1.contains_point(2.1, 5)
    assert not seg1.contains_point(2, 4.9)

    seg2 = LineSegment(x1=2, y1=-5, x2=1, y2=6)
    seg2.plot(ax, color="blue", linewidth=2)

    seg3 = LineSegment(x1=-2, y1=-3, x2=-4, y2=2)
    seg3.plot(ax, color="green", linewidth=2)

    assert geom2ds_intersect(seg1, seg2)
    assert not geom2ds_intersect(seg1, seg3)
    assert not geom2ds_intersect(seg2, seg3)

    rng = np.random.default_rng(0)
    for _ in range(10):
        p1 = seg1.sample_random_point(rng)
        assert seg1.contains_point(p1[0], p1[1])
        plt.plot(p1[0], p1[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/line_segment_unit_test.png")

    # Legacy tests.
    seg1 = LineSegment(2, 5, 7, 6)
    seg2 = LineSegment(2.5, 7.1, 7.4, 5.3)
    assert geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(1, 3, 5, 3)
    seg2 = LineSegment(3, 7, 3, 2)
    assert geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(2, 5, 7, 6)
    seg2 = LineSegment(2, 6, 7, 7)
    assert not geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(1, 1, 3, 3)
    seg2 = LineSegment(2, 2, 4, 4)
    assert not geom2ds_intersect(seg1, seg2)

    seg1 = LineSegment(1, 1, 3, 3)
    seg2 = LineSegment(1, 1, 6.7, 7.4)
    assert not geom2ds_intersect(seg1, seg2)


def test_circle():
    """Tests for Circle()."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim((-11, 5))
    ax.set_ylim((-6, 10))

    circ1 = Circle(x=0, y=1, radius=3)
    assert circ1.x == 0
    assert circ1.y == 1
    assert circ1.radius == 3
    circ1.plot(ax, color="red", alpha=0.5)

    assert circ1.contains_point(0, 1)
    assert circ1.contains_point(0.5, 1)
    assert circ1.contains_point(0, 0.5)
    assert circ1.contains_point(0.25, 1.25)
    assert not circ1.contains_point(0, 4.1)
    assert not circ1.contains_point(3.1, 0)
    assert not circ1.contains_point(0, -2.1)
    assert not circ1.contains_point(-3.1, 0)

    circ2 = Circle(x=-3, y=2, radius=6)
    circ2.plot(ax, color="blue", alpha=0.5)

    circ3 = Circle(x=-6, y=1, radius=1)
    circ3.plot(ax, color="green", alpha=0.5)

    assert geom2ds_intersect(circ1, circ2)
    assert not geom2ds_intersect(circ1, circ3)
    assert geom2ds_intersect(circ2, circ3)

    rng = np.random.default_rng(0)
    for _ in range(10):
        p3 = circ3.sample_random_point(rng)
        assert circ3.contains_point(p3[0], p3[1])
        plt.plot(p3[0], p3[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/circle_unit_test.png")


def test_triangle():
    """Tests for Triangle()."""
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim((-10.0, 10.0))
    ax.set_ylim((-10.0, 10.0))

    tri1 = Triangle(5.0, 5.0, 7.5, 7.5, 5.0, 7.5)
    assert tri1.contains_point(5.5, 6)
    assert tri1.contains_point(5.9999, 6)
    assert tri1.contains_point(5.8333, 6.6667)
    assert tri1.contains_point(7.3, 7.4)
    assert not tri1.contains_point(6, 6)
    assert not tri1.contains_point(5.1, 5.1)
    assert not tri1.contains_point(5.2, 5.1)
    assert not tri1.contains_point(5.1, 7.6)
    assert not tri1.contains_point(4.9, 7.3)
    assert not tri1.contains_point(5.0, 7.5)
    assert not tri1.contains_point(7.6, 7.6)
    tri1.plot(ax, color="red", alpha=0.5)

    tri2 = Triangle(-3.0, -4.0, -6.2, -5.6, -9.0, -1.7)
    tri2.plot(ax, color="blue", alpha=0.5)

    # Almost degenerate triangle.
    tri3 = Triangle(0.0, 0.0, 1.0, 1.0, -1.0, -1.001)
    assert tri3.contains_point(0.0, -0.001 / 3.0)
    tri3.plot(ax, color="green", alpha=0.5)

    # Degenerate triangle (a line).
    with pytest.raises(ValueError) as e:
        Triangle(0.0, 0.0, 1.0, 1.0, -1.0, -1.0)
    assert "Degenerate triangle" in str(e)

    rng = np.random.default_rng(0)
    for _ in range(10):
        p1 = tri1.sample_random_point(rng)
        assert tri1.contains_point(p1[0], p1[1])
        plt.plot(p1[0], p1[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/triangle_unit_test.png")


def test_rectangle():
    """Tests for Rectangle()."""
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    rect1 = Rectangle(x=-2, y=-1, width=4, height=3, theta=0)
    assert rect1.x == -2
    assert rect1.y == -1
    assert rect1.width == 4
    assert rect1.height == 3
    assert rect1.theta == 0
    rect1.plot(ax, color="red", alpha=0.5)

    assert np.allclose(rect1.center, (0, 0.5))

    circ1 = rect1.circumscribed_circle
    assert np.allclose((circ1.x, circ1.y), (0, 0.5))
    assert np.allclose(circ1.radius, 2.5)
    circ1.plot(ax, facecolor="none", edgecolor="black", linewidth=1, linestyle="dashed")

    expected_vertices = np.array([(-2, -1), (-2, 2), (2, -1), (2, 2)])
    assert np.allclose(sorted(rect1.vertices), expected_vertices)
    for x, y in rect1.vertices:
        v = Circle(x, y, radius=0.1)
        v.plot(ax, facecolor="none", edgecolor="black", linewidth=1, linestyle="dashed")

    for seg in rect1.line_segments:
        seg.plot(ax, color="black", linewidth=1, linestyle="dashed")

    assert not rect1.contains_point(-2.1, 0)
    assert rect1.contains_point(-1.9, 0)
    assert not rect1.contains_point(0, 2.1)
    assert rect1.contains_point(0, 1.9)
    assert not rect1.contains_point(2.1, 0)
    assert rect1.contains_point(1.9, 0)
    assert not rect1.contains_point(0, -1.1)
    assert rect1.contains_point(0, -0.9)
    assert rect1.contains_point(0, 0.5)
    assert not rect1.contains_point(100, 100)

    rect2 = Rectangle(x=1, y=-2, width=2, height=2, theta=0.5)
    rect2.plot(ax, color="blue", alpha=0.5)

    rect3 = Rectangle(x=-1.5, y=1, width=1, height=1, theta=-0.5)
    rect3.plot(ax, color="green", alpha=0.5)

    assert geom2ds_intersect(rect1, rect2)
    assert geom2ds_intersect(rect1, rect3)
    assert geom2ds_intersect(rect3, rect1)
    assert not geom2ds_intersect(rect2, rect3)

    rect4 = Rectangle(x=0.8, y=1e-5, height=0.1, width=0.07, theta=0)
    assert not rect4.contains_point(0.2, 0.05)

    rect5 = Rectangle(x=-4, y=-2, height=0.25, width=2, theta=-np.pi / 4)
    rect5.plot(ax, facecolor="yellow", edgecolor="gray")
    origin = Circle(x=-3.5, y=-2.3, radius=0.05)
    origin.plot(ax, color="black")
    rect6 = rect5.rotate_about_point(origin.x, origin.y, rot=np.pi / 4)
    rect6.plot(ax, facecolor="none", edgecolor="black", linestyle="dashed")

    rect7 = Rectangle.from_center(
        center_x=1, center_y=2, width=2, height=4, rotation_about_center=0
    )
    rect7.plot(ax, facecolor="grey")
    assert rect7.center == (1, 2)

    rng = np.random.default_rng(0)
    for _ in range(100):
        p5 = rect5.sample_random_point(rng)
        assert rect5.contains_point(p5[0], p5[1])
        plt.plot(p5[0], p5[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/rectangle_unit_test.png")

    # Tests scale_about_center().
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    rect8 = Rectangle(x=-2, y=-1, width=4, height=3, theta=0)
    rect8.plot(ax, color="red", alpha=0.5)
    rect9 = rect8.scale_about_center(width_scale=0.75, height_scale=0.5)
    assert set(rect9.vertices) == {
        (-1.5, -0.25),
        (-1.5, 1.25),
        (1.5, 1.25),
        (1.5, -0.25),
    }
    rect9.plot(ax, color="blue", alpha=0.5)

    # Uncomment for debugging.
    # plt.savefig("/tmp/rectangle_unit_test2.png")


def test_lobject():
    """Tests for Lobject."""
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    lobject = Lobject(x=3, y=4, width=0.5, lengths=(2, 3), theta=0.0)

    assert lobject.x == 3
    assert lobject.y == 4
    assert lobject.width == 0.5
    assert lobject.lengths[0] == 2
    assert lobject.lengths[1] == 3
    assert lobject.theta == 0.0

    lobject.plot(ax, color="purple", alpha=0.5)

    expected_vertices = np.array(
        [(3, 4), (1, 4), (1, 3.5), (2.5, 3.5), (2.5, 1), (3, 1), (3, 3.5), (2.5, 4)]
    )
    np.testing.assert_array_equal(lobject.vertices, expected_vertices)

    # Test rotation about center
    lobject = lobject.rotate_about_point(lobject.x, lobject.y, np.pi / 6)
    lobject.plot(ax, color="orange", alpha=0.5)

    # Test rotation about external point
    lobject = lobject.rotate_about_point(0, 0, np.pi / 6)
    lobject.plot(ax, color="red", alpha=0.5)

    # Test scaling about center
    lobject = lobject.scale_about_center(width_scale=0.5, length_scale=0.5)
    lobject.plot(ax, color="blue", alpha=0.5)

    # Test sample_random_point
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = lobject.sample_random_point(rng)
        assert lobject.contains_point(p[0], p[1])
        plt.plot(p[0], p[1], "bo")

    # Uncomment for debugging.
    # plt.savefig("/tmp/lobject_unit_test.png")


def test_line_segment_circle_intersection():
    """Tests for line_segment_intersects_circle()."""
    seg1 = LineSegment(-3, 0, 0, 0)
    circ1 = Circle(0, 0, 1)
    assert geom2ds_intersect(seg1, circ1)
    assert geom2ds_intersect(circ1, seg1)

    seg2 = LineSegment(-3, 3, 4, 3)
    assert not geom2ds_intersect(seg2, circ1)
    assert not geom2ds_intersect(circ1, seg2)

    seg3 = LineSegment(0, -2, 1, -2.5)
    assert not geom2ds_intersect(seg3, circ1)
    assert not geom2ds_intersect(circ1, seg3)

    seg4 = LineSegment(0, -3, 0, -4)
    assert not geom2ds_intersect(seg4, circ1)
    assert not geom2ds_intersect(circ1, seg4)
    assert not geom2ds_intersect(seg2, circ1)

    # Uncomment for debugging.
    # plt.savefig("/tmp/line_segment_circle_unit_test.png")


def test_line_segment_rectangle_intersection():
    """Tests for line_segment_intersects_rectangle()."""
    seg1 = LineSegment(-3, 0, 0, 0)
    rect1 = Rectangle(-1, -1, 2, 2, 0)
    assert geom2ds_intersect(seg1, rect1)
    assert geom2ds_intersect(rect1, seg1)

    seg2 = LineSegment(-3, 3, 4, 3)
    assert not geom2ds_intersect(seg2, rect1)
    assert not geom2ds_intersect(rect1, seg2)

    seg3 = LineSegment(0, -2, 1, -2.5)
    assert not geom2ds_intersect(seg3, rect1)
    assert not geom2ds_intersect(rect1, seg3)

    seg4 = LineSegment(0, -3, 0, -4)
    assert not geom2ds_intersect(seg4, rect1)
    assert not geom2ds_intersect(rect1, seg4)


def test_rectangle_circle_intersection():
    """Tests for rectangle_intersects_circle()."""
    rect1 = Rectangle(x=0, y=0, width=4, height=3, theta=0)
    circ1 = Circle(x=0, y=0, radius=1)
    assert geom2ds_intersect(rect1, circ1)
    assert geom2ds_intersect(circ1, rect1)

    circ2 = Circle(x=1, y=1, radius=0.5)
    assert geom2ds_intersect(rect1, circ2)
    assert geom2ds_intersect(circ2, rect1)

    rect2 = Rectangle(x=1, y=1, width=1, height=1, theta=0)
    assert not geom2ds_intersect(rect2, circ1)
    assert not geom2ds_intersect(circ1, rect2)

    circ3 = Circle(x=0, y=0, radius=100)
    assert geom2ds_intersect(rect1, circ3)
    assert geom2ds_intersect(circ3, rect1)
    assert geom2ds_intersect(rect2, circ3)
    assert geom2ds_intersect(circ3, rect2)


def test_lobject_rectangle_intersection():
    """Tests for Lobject intersection."""
    lobject = Lobject(x=0, y=0, width=1, lengths=[1, 1], theta=0)
    assert geom2ds_intersect(lobject, Rectangle(x=0, y=0, width=1, height=1, theta=0))
    assert geom2ds_intersect(Rectangle(x=0, y=0, width=1, height=1, theta=0), lobject)


def test_lobject_circle_intersection():
    """Tests for Lobject intersection."""
    lobject = Lobject(x=0, y=0, width=1, lengths=[1, 1], theta=0)
    assert geom2ds_intersect(lobject, Circle(x=0, y=0, radius=1))
    assert geom2ds_intersect(Circle(x=0, y=0, radius=1), lobject)


def test_rtrapezoid():
    """Tests for RTrapezoid()."""
    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim((-5, 5))
    ax.set_ylim((-5, 5))

    # Create basic trapezoid: l=4, h=2
    # Vertices: (0,0), (4,0), (2,2), (0,2)
    trap1 = RTrapezoid(x=0, y=0, l=4, h=2, theta=0)
    assert trap1.x == 0
    assert trap1.y == 0
    assert trap1.l == 4
    assert trap1.h == 2
    assert trap1.theta == 0
    trap1.plot(ax, color="red", alpha=0.5)

    # Check vertices
    expected_vertices = [(0, 0), (4, 0), (2, 2), (0, 2)]
    assert np.allclose(sorted(trap1.vertices), sorted(expected_vertices))

    # Plot vertices
    for x, y in trap1.vertices:
        v = Circle(x, y, radius=0.1)
        v.plot(ax, facecolor="none", edgecolor="black", linewidth=1, linestyle="dashed")

    # Plot line segments
    for seg in trap1.line_segments:
        seg.plot(ax, color="black", linewidth=1, linestyle="dashed")

    # Test contains_point
    assert trap1.contains_point(1, 1)  # Inside the rectangular part
    assert trap1.contains_point(0.5, 0.5)  # Inside the rectangular part
    assert trap1.contains_point(2.5, 0.5)  # In the triangular part
    assert trap1.contains_point(3, 1)  # In the triangular part
    assert not trap1.contains_point(3.5, 1.5)  # Outside (above slant)
    assert not trap1.contains_point(-0.1, 1)  # Outside (left)
    assert not trap1.contains_point(4.1, 0.5)  # Outside (right)
    assert not trap1.contains_point(1, 2.1)  # Outside (top)
    assert not trap1.contains_point(1, -0.1)  # Outside (bottom)

    # Test area
    expected_area = (4 + 2) * 2 / 2  # (l + (l-h)) * h / 2 = 6
    assert np.isclose(trap1.area, expected_area)

    # Test center
    center = trap1.center
    assert center[0] > 0 and center[0] < 4
    assert center[1] > 0 and center[1] < 2
    center_circle = Circle(center[0], center[1], radius=0.1)
    center_circle.plot(ax, color="blue")

    # Create rotated trapezoid
    trap2 = RTrapezoid(x=1, y=-2, l=3, h=1.5, theta=0.5)
    trap2.plot(ax, color="blue", alpha=0.5)

    # Create another trapezoid
    trap3 = RTrapezoid(x=-3, y=1, l=2.5, h=1, theta=-0.3)
    trap3.plot(ax, color="green", alpha=0.5)

    # Test intersections
    assert geom2ds_intersect(trap1, trap2)
    assert not geom2ds_intersect(trap1, trap3)
    assert not geom2ds_intersect(trap2, trap3)

    # Test validation - h must be < l
    with pytest.raises(ValueError) as e:
        RTrapezoid(x=0, y=0, l=2, h=2, theta=0)
    assert "Height h must be less than length l" in str(e)

    with pytest.raises(ValueError) as e:
        RTrapezoid(x=0, y=0, l=2, h=3, theta=0)
    assert "Height h must be less than length l" in str(e)

    # Test negative values
    with pytest.raises(ValueError) as e:
        RTrapezoid(x=0, y=0, l=-2, h=1, theta=0)
    assert "Length l must be positive" in str(e)

    with pytest.raises(ValueError) as e:
        RTrapezoid(x=0, y=0, l=2, h=-1, theta=0)
    assert "Height h must be positive" in str(e)

    # Test sample_random_point
    rng = np.random.default_rng(0)
    for _ in range(100):
        p = trap1.sample_random_point(rng)
        assert trap1.contains_point(p[0], p[1])
        plt.plot(p[0], p[1], "ro", markersize=2)

    # Test rotate_about_point
    trap4 = RTrapezoid(x=-4, y=-2, l=2, h=1, theta=0)
    trap4.plot(ax, facecolor="yellow", edgecolor="gray")
    origin = Circle(x=-3, y=-1.5, radius=0.05)
    origin.plot(ax, color="black")
    trap5 = trap4.rotate_about_point(origin.x, origin.y, rot=np.pi / 4)
    trap5.plot(ax, facecolor="none", edgecolor="black", linestyle="dashed")

    # Test scale_about_center
    trap6 = RTrapezoid(x=2, y=2, l=2, h=1, theta=0.2)
    trap6.plot(ax, facecolor="purple", alpha=0.3)
    trap7 = trap6.scale_about_center(scale=0.5)
    trap7.plot(ax, facecolor="orange", alpha=0.5)
    # Verify centers match
    assert np.allclose(trap6.center, trap7.center, atol=0.1)

    # Uncomment for debugging.
    # plt.savefig("/tmp/rtrapezoid_unit_test.png")


def test_rtrapezoid_line_segment_intersection():
    """Tests for line_segment_intersects_rtrapezoid()."""
    trap1 = RTrapezoid(x=0, y=0, l=4, h=2, theta=0)
    seg1 = LineSegment(-1, 1, 2, 1)
    assert geom2ds_intersect(seg1, trap1)
    assert geom2ds_intersect(trap1, seg1)

    seg2 = LineSegment(-2, 3, 5, 3)
    assert not geom2ds_intersect(seg2, trap1)
    assert not geom2ds_intersect(trap1, seg2)

    seg3 = LineSegment(1, -2, 1, -1)
    assert not geom2ds_intersect(seg3, trap1)
    assert not geom2ds_intersect(trap1, seg3)


def test_rtrapezoid_circle_intersection():
    """Tests for rtrapezoid_intersects_circle()."""
    trap1 = RTrapezoid(x=0, y=0, l=4, h=2, theta=0)
    circ1 = Circle(x=1, y=1, radius=0.5)
    assert geom2ds_intersect(trap1, circ1)
    assert geom2ds_intersect(circ1, trap1)

    circ2 = Circle(x=5, y=1, radius=0.5)
    assert not geom2ds_intersect(trap1, circ2)
    assert not geom2ds_intersect(circ2, trap1)

    circ3 = Circle(x=2, y=2, radius=3)
    assert geom2ds_intersect(trap1, circ3)
    assert geom2ds_intersect(circ3, trap1)


def test_rtrapezoid_rectangle_intersection():
    """Tests for rtrapezoid_intersects_rectangle()."""
    trap1 = RTrapezoid(x=0, y=0, l=4, h=2, theta=0)
    rect1 = Rectangle(x=1, y=1, width=1, height=1, theta=0)
    assert geom2ds_intersect(trap1, rect1)
    assert geom2ds_intersect(rect1, trap1)

    rect2 = Rectangle(x=5, y=1, width=1, height=1, theta=0)
    assert not geom2ds_intersect(trap1, rect2)
    assert not geom2ds_intersect(rect2, trap1)

    rect3 = Rectangle(x=-1, y=-1, width=6, height=4, theta=0)
    assert geom2ds_intersect(trap1, rect3)
    assert geom2ds_intersect(rect3, trap1)


def test_triangle_line_segment_intersection():
    """Tests for line_segment_intersects_triangle()."""
    tri1 = Triangle(0, 0, 4, 0, 2, 3)

    # Line segment that intersects the triangle
    seg1 = LineSegment(1, 1, 3, 1)
    assert geom2ds_intersect(seg1, tri1)
    assert geom2ds_intersect(tri1, seg1)

    # Line segment that passes through a vertex
    seg2 = LineSegment(-1, -1, 1, 1)
    assert geom2ds_intersect(seg2, tri1)
    assert geom2ds_intersect(tri1, seg2)

    # Line segment completely inside the triangle
    seg3 = LineSegment(2, 1, 2.5, 1.5)
    assert geom2ds_intersect(seg3, tri1)
    assert geom2ds_intersect(tri1, seg3)

    # Line segment completely outside the triangle
    seg4 = LineSegment(5, 5, 6, 6)
    assert not geom2ds_intersect(seg4, tri1)
    assert not geom2ds_intersect(tri1, seg4)

    # Line segment that crosses an edge
    seg5 = LineSegment(0, 1, 4, 1)
    assert geom2ds_intersect(seg5, tri1)
    assert geom2ds_intersect(tri1, seg5)


def test_triangle_circle_intersection():
    """Tests for triangle_intersects_circle()."""
    tri1 = Triangle(0, 0, 4, 0, 2, 3)

    # Circle centered inside triangle
    circ1 = Circle(x=2, y=1, radius=0.5)
    assert geom2ds_intersect(tri1, circ1)
    assert geom2ds_intersect(circ1, tri1)

    # Circle that overlaps with triangle edge
    circ2 = Circle(x=0, y=0, radius=1)
    assert geom2ds_intersect(tri1, circ2)
    assert geom2ds_intersect(circ2, tri1)

    # Circle completely outside triangle
    circ3 = Circle(x=10, y=10, radius=1)
    assert not geom2ds_intersect(tri1, circ3)
    assert not geom2ds_intersect(circ3, tri1)

    # Circle that contains a vertex
    circ4 = Circle(x=2, y=3, radius=0.5)
    assert geom2ds_intersect(tri1, circ4)
    assert geom2ds_intersect(circ4, tri1)

    # Large circle that contains the triangle
    circ5 = Circle(x=2, y=1.5, radius=10)
    assert geom2ds_intersect(tri1, circ5)
    assert geom2ds_intersect(circ5, tri1)


def test_triangle_rectangle_intersection():
    """Tests for triangle_intersects_rectangle()."""
    tri1 = Triangle(0, 0, 4, 0, 2, 3)

    # Rectangle that overlaps with triangle
    rect1 = Rectangle(x=1, y=0.5, width=2, height=1, theta=0)
    assert geom2ds_intersect(tri1, rect1)
    assert geom2ds_intersect(rect1, tri1)

    # Rectangle completely inside triangle
    rect2 = Rectangle(x=1.5, y=0.5, width=1, height=0.5, theta=0)
    assert geom2ds_intersect(tri1, rect2)
    assert geom2ds_intersect(rect2, tri1)

    # Rectangle completely outside triangle
    rect3 = Rectangle(x=5, y=5, width=1, height=1, theta=0)
    assert not geom2ds_intersect(tri1, rect3)
    assert not geom2ds_intersect(rect3, tri1)

    # Rotated rectangle that intersects triangle
    rect4 = Rectangle(x=1, y=1, width=2, height=1, theta=0.5)
    assert geom2ds_intersect(tri1, rect4)
    assert geom2ds_intersect(rect4, tri1)

    # Rectangle that contains the triangle
    rect5 = Rectangle(x=-1, y=-1, width=6, height=5, theta=0)
    assert geom2ds_intersect(tri1, rect5)
    assert geom2ds_intersect(rect5, tri1)


def test_triangle_triangle_intersection():
    """Tests for triangles_intersect()."""
    tri1 = Triangle(0, 0, 4, 0, 2, 3)

    # Overlapping triangles
    tri2 = Triangle(1, 1, 5, 1, 3, 4)
    assert geom2ds_intersect(tri1, tri2)
    assert geom2ds_intersect(tri2, tri1)

    # Triangles sharing a vertex
    tri3 = Triangle(0, 0, -2, 0, -1, 2)
    assert geom2ds_intersect(tri1, tri3)
    assert geom2ds_intersect(tri3, tri1)

    # Non-intersecting triangles
    tri4 = Triangle(10, 10, 12, 10, 11, 12)
    assert not geom2ds_intersect(tri1, tri4)
    assert not geom2ds_intersect(tri4, tri1)

    # One triangle inside another
    tri5 = Triangle(1.5, 0.5, 2.5, 0.5, 2, 1.5)
    assert geom2ds_intersect(tri1, tri5)
    assert geom2ds_intersect(tri5, tri1)

    # Triangles with overlapping edges (not just touching at a point)
    tri6 = Triangle(1, 2, 3, 2, 2, 4)
    assert geom2ds_intersect(tri1, tri6)
    assert geom2ds_intersect(tri6, tri1)


def test_triangle_rtrapezoid_intersection():
    """Tests for triangle_intersects_rtrapezoid()."""
    tri1 = Triangle(0, 0, 4, 0, 2, 3)
    trap1 = RTrapezoid(x=1, y=0.5, l=3, h=2, theta=0)

    # Triangle and trapezoid that overlap
    assert geom2ds_intersect(tri1, trap1)
    assert geom2ds_intersect(trap1, tri1)

    # Non-intersecting triangle and trapezoid
    trap2 = RTrapezoid(x=10, y=10, l=3, h=2, theta=0)
    assert not geom2ds_intersect(tri1, trap2)
    assert not geom2ds_intersect(trap2, tri1)

    # Triangle inside trapezoid
    tri2 = Triangle(1.2, 0.6, 2, 0.6, 1.6, 1.2)
    assert geom2ds_intersect(tri2, trap1)
    assert geom2ds_intersect(trap1, tri2)

    # Rotated trapezoid intersecting triangle
    trap3 = RTrapezoid(x=1, y=1, l=2, h=1, theta=0.3)
    assert geom2ds_intersect(tri1, trap3)
    assert geom2ds_intersect(trap3, tri1)


def test_triangle_lobject_intersection():
    """Tests for triangle_intersects_lobject()."""
    tri1 = Triangle(0, 0, 4, 0, 2, 3)
    lobj1 = Lobject(x=2, y=1, width=0.5, lengths=(1.5, 2), theta=0)

    # Triangle and L-object that overlap
    assert geom2ds_intersect(tri1, lobj1)
    assert geom2ds_intersect(lobj1, tri1)

    # Non-intersecting triangle and L-object
    lobj2 = Lobject(x=10, y=10, width=0.5, lengths=(1, 1), theta=0)
    assert not geom2ds_intersect(tri1, lobj2)
    assert not geom2ds_intersect(lobj2, tri1)

    # L-object intersecting triangle edge
    lobj3 = Lobject(x=0, y=0, width=0.3, lengths=(1, 1), theta=0)
    assert geom2ds_intersect(tri1, lobj3)
    assert geom2ds_intersect(lobj3, tri1)

    # Rotated L-object intersecting triangle
    lobj4 = Lobject(x=2, y=1.5, width=0.4, lengths=(1, 1.5), theta=0.5)
    assert geom2ds_intersect(tri1, lobj4)
    assert geom2ds_intersect(lobj4, tri1)


def test_geom2ds_intersect():
    """Tests for geom2ds_intersect()."""
    with pytest.raises(NotImplementedError):
        geom2ds_intersect(None, None)
