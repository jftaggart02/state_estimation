"""Dubins path planning for planar robots.

This module implements the six elementary Dubins path types and transforms
paths from a start-relative coordinate system into a global coordinate system.
It also provides plotting helpers for visualizing generated paths.

This code was adapted from the original Python implementation by Atsushi Sakai (@Atsushi_twi), which is
available at https://github.com/AtsushiSakai/PythonRobotics/.
"""

from math import sin, cos, atan2, sqrt, acos, pi, hypot
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def plot_arrow(
    x: Union[float, Sequence[float], np.ndarray],
    y: Union[float, Sequence[float], np.ndarray],
    yaw: Union[float, Sequence[float], np.ndarray],
    arrow_length: float = 1.0,
    origin_point_plot_style: Optional[str] = "xr",
    head_width: float = 0.1,
    fc: str = "r",
    ec: str = "k",
    **kwargs: Any
) -> None:
    """Plot one or more arrows from 2D position and yaw values.

    Args:
        x: Arrow origin x-position or a sequence of x-positions.
        y: Arrow origin y-position or a sequence of y-positions.
        yaw: Arrow yaw angle in radians or a sequence of yaw angles.
        arrow_length: Length of each arrow.
        origin_point_plot_style: Matplotlib plot style for origin points.
            Set to ``None`` to omit origin points.
        head_width: Width of each arrow head.
        fc: Arrow face color.
        ec: Arrow edge color.
        **kwargs: Additional keyword arguments passed to
            :meth:`matplotlib.pyplot.arrow`.
    """
    # Plot a sequence recursively so each arrow uses the same style options.
    if not isinstance(x, float):
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, head_width=head_width, fc=fc, ec=ec, **kwargs)
    else:
        # Draw the arrow from its origin, then optionally mark the origin point.
        plt.arrow(x, y, arrow_length * cos(yaw), arrow_length * sin(yaw), head_width=head_width, fc=fc, ec=ec, **kwargs)
        if origin_point_plot_style is not None:
            plt.plot(x, y, origin_point_plot_style)


def rot_mat_2d(angle: float) -> np.ndarray:
    """Create a 2D rotation matrix for an angle.

    Args:
        angle: Rotation angle in radians.

    Returns:
        The 2-by-2 rotation matrix.
    """
    return Rot.from_euler("z", angle).as_matrix()[0:2, 0:2]


def angle_mod(
    x: Union[float, Sequence[float], np.ndarray], zero_2_2pi: bool = False, degree: bool = False
) -> Union[float, np.ndarray]:
    """Normalize angles into a signed or full-circle range.

    By default, results are in the range ``[-pi, pi)``. When
    ``zero_2_2pi`` is true, results are in ``[0, 2pi)``.

    Args:
        x: An angle or an array-like collection of angles.
        zero_2_2pi: Use ``[0, 2pi)`` instead of ``[-pi, pi)``.
        degree: Treat input and return values as degrees.

    Returns:
        A float for a float input, or a flattened NumPy array otherwise.
    """
    # Preserve the scalar/array return convention before converting the input.
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    # Flatten array-like input and convert degree values to radians if needed.
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    # Apply the requested modulo range.
    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    # Restore degree units and the original scalar/array return shape.
    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


def plan_dubins_path(
    s_x: float,
    s_y: float,
    s_yaw: float,
    g_x: float,
    g_y: float,
    g_yaw: float,
    curvature: float,
    step_size: float = 0.1,
    selected_types: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[float]]:
    """Plan the shortest feasible Dubins path between two poses.

    Args:
        s_x: Start-point x-coordinate in meters.
        s_y: Start-point y-coordinate in meters.
        s_yaw: Start-point yaw angle in radians.
        g_x: Goal-point x-coordinate in meters.
        g_y: Goal-point y-coordinate in meters.
        g_yaw: Goal-point yaw angle in radians.
        curvature: Maximum path curvature in inverse meters.
        step_size: Distance between interpolated path points in meters.
        selected_types: Optional sequence of elementary path types to consider,
            such as ``["RSL", "RSR"]``. Uses all types when omitted.

    Returns:
        x_list: Global x-coordinates of the path.
        y_list: Global y-coordinates of the path.
        yaw_list: Global yaw angles of the path in radians.
        modes: Names of the path segments in order.
        lengths: Physical lengths of the path segments in meters.
    """
    # Limit candidate search to the requested path types, or use every type.
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # Transform the goal pose into the start pose's local coordinate frame.
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    # Calculate the shortest local Dubins path for the transformed goal.
    lp_x, lp_y, lp_yaw, modes, lengths = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size, planning_funcs
    )

    # Rotate the local path back and translate it into global coordinates.
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    return x_list, y_list, yaw_list, modes, lengths


def _mod2pi(theta: float) -> float:
    """Normalize an angle to the range ``[0, 2pi)``.

    Args:
        theta: Angle in radians.

    Returns:
        The normalized angle.
    """
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha: float, beta: float) -> Tuple[float, float, float, float, float]:
    """Calculate trigonometric values shared by Dubins path formulas.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.

    Returns:
        sin_a: Sine of the start heading angle.
        sin_b: Sine of the goal heading angle.
        cos_a: Cosine of the start heading angle.
        cos_b: Cosine of the goal heading angle.
        cos_ab: Cosine of the difference between the heading angles.
    """
    # Compute all trigonometric terms used across the elementary path formulas.
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha: float, beta: float, d: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """Calculate an LSL Dubins path candidate.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.
        d: Normalized distance between poses.

    Returns:
        d1: Length of the first segment, or ``None`` if infeasible.
        d2: Length of the second segment, or ``None`` if infeasible.
        d3: Length of the third segment, or ``None`` if infeasible.
        mode: Names of the path segments in order.
    """
    # Define the segment pattern and derive its feasibility equation.
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d**2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    # Normalize the computed turn lengths and straight length.
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha: float, beta: float, d: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """Calculate an RSR Dubins path candidate.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.
        d: Normalized distance between poses.

    Returns:
        d1: Length of the first segment, or ``None`` if infeasible.
        d2: Length of the second segment, or ``None`` if infeasible.
        d3: Length of the third segment, or ``None`` if infeasible.
        mode: Names of the path segments in order.
    """
    # Define the segment pattern and derive its feasibility equation.
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d**2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    # Normalize the computed turn lengths and straight length.
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha: float, beta: float, d: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """Calculate an LSR Dubins path candidate.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.
        d: Normalized distance between poses.

    Returns:
        d1: Length of the first returned segment, or ``None`` if infeasible.
        d2: Length of the second returned segment, or ``None`` if infeasible.
        d3: Length of the third returned segment, or ``None`` if infeasible.
        mode: Names of the path segments in order.
    """
    # Define the segment pattern and reject configurations with no real solution.
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d**2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    # Derive and normalize the ordered segment lengths.
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha: float, beta: float, d: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """Calculate an RSL Dubins path candidate.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.
        d: Normalized distance between poses.

    Returns:
        d1: Length of the first returned segment, or ``None`` if infeasible.
        d2: Length of the second returned segment, or ``None`` if infeasible.
        d3: Length of the third returned segment, or ``None`` if infeasible.
        mode: Names of the path segments in order.
    """
    # Define the segment pattern and reject configurations with no real solution.
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d**2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    # Derive and normalize the ordered segment lengths.
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha: float, beta: float, d: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """Calculate an RLR Dubins path candidate.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.
        d: Normalized distance between poses.

    Returns:
        d1: Length of the first segment, or ``None`` if infeasible.
        d2: Length of the second segment, or ``None`` if infeasible.
        d3: Length of the third segment, or ``None`` if infeasible.
        mode: Names of the path segments in order.
    """
    # Define the segment pattern and test whether the middle arc is valid.
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d**2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    # Compute and normalize the alternating turn lengths.
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha: float, beta: float, d: float) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """Calculate an LRL Dubins path candidate.

    Args:
        alpha: Start heading angle.
        beta: Goal heading angle.
        d: Normalized distance between poses.

    Returns:
        d1: Length of the first segment, or ``None`` if infeasible.
        d2: Length of the second segment, or ``None`` if infeasible.
        d3: Length of the third segment, or ``None`` if infeasible.
        mode: Names of the path segments in order.
    """
    # Define the segment pattern and test whether the middle arc is valid.
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d**2 + 2.0 * cos_ab + 2.0 * d * (-sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    # Compute and normalize the alternating turn lengths.
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {
    "LSL": _LSL,
    "RSR": _RSR,
    "LSR": _LSR,
    "RSL": _RSL,
    "RLR": _RLR,
    "LRL": _LRL,
}


def _dubins_path_planning_from_origin(
    end_x: float,
    end_y: float,
    end_yaw: float,
    curvature: float,
    step_size: float,
    planning_funcs: Iterable[
        Callable[[float, float, float], Tuple[Optional[float], Optional[float], Optional[float], List[str]]]
    ],
) -> Tuple[List[float], List[float], List[float], List[str], List[float]]:
    """Select and generate the shortest Dubins path from the origin.

    Args:
        end_x: Goal x-coordinate in the origin-relative frame.
        end_y: Goal y-coordinate in the origin-relative frame.
        end_yaw: Goal yaw angle in radians.
        curvature: Maximum path curvature in inverse meters.
        step_size: Distance between interpolated path points in meters.
        planning_funcs: Dubins path candidate functions to evaluate.

    Returns:
        x_list: Local x-coordinates of the path.
        y_list: Local y-coordinates of the path.
        yaw_list: Local yaw angles of the path in radians.
        modes: Names of the path segments in order.
        lengths: Physical lengths of the path segments in meters.
    """
    # Convert the Euclidean goal into normalized heading and distance inputs.
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    # Track the lowest-cost feasible candidate across all selected path types.
    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue

        # Replace the best candidate whenever this path has a shorter total cost.
        cost = abs(d1) + abs(d2) + abs(d3)
        if best_cost > cost:  # Select minimum length one.
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

    # Generate normalized local points, then convert lengths to physical units.
    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode, curvature, step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, b_mode, lengths


def _interpolate(
    length: float,
    mode: str,
    max_curvature: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
    path_x: List[float],
    path_y: List[float],
    path_yaw: List[float],
) -> Tuple[List[float], List[float], List[float]]:
    """Append one interpolated Dubins path point.

    Args:
        length: Path distance along the current segment.
        mode: Segment mode, either ``"L"``, ``"S"``, or ``"R"``.
        max_curvature: Maximum path curvature in inverse meters.
        origin_x: Segment origin x-coordinate.
        origin_y: Segment origin y-coordinate.
        origin_yaw: Segment origin yaw angle.
        path_x: Accumulated x-coordinates.
        path_y: Accumulated y-coordinates.
        path_yaw: Accumulated yaw angles.

    Returns:
        path_x: Updated x-coordinates of the path.
        path_y: Updated y-coordinates of the path.
        path_yaw: Updated yaw angles of the path in radians.
    """
    # Straight segments extend directly along the current yaw.
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        # Compute the segment displacement in the segment-local frame.
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        # Rotate the local displacement into the global frame and append it.
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        # Update yaw according to whether the segment turns left or right.
        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(
    lengths: Sequence[float], modes: Sequence[str], max_curvature: float, step_size: float
) -> Tuple[List[float], List[float], List[float]]:
    """Generate uniformly sampled points for a local Dubins path.

    Args:
        lengths: Physical lengths of the path segments.
        modes: Segment modes in path order.
        max_curvature: Maximum path curvature in inverse meters.
        step_size: Distance between sampled path points.

    Returns:
        p_x: Local x-coordinates of the path.
        p_y: Local y-coordinates of the path.
        p_yaw: Local yaw angles of the path in radians.
    """
    # Start at the origin and append sampled states for each segment in order.
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

    for mode, length in zip(modes, lengths):
        if length == 0.0:
            continue

        # Capture the end of the preceding segment as this segment's origin.
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]

        # Sample interior points at the requested step size.
        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature, origin_x, origin_y, origin_yaw, p_x, p_y, p_yaw)
            current_length += step_size

        # Add the exact segment endpoint to avoid a remaining length gap.
        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x, origin_y, origin_yaw, p_x, p_y, p_yaw)

    return p_x, p_y, p_yaw


def main() -> None:
    """Run a sample Dubins path plan and display its plot."""
    # Define a representative start pose, goal pose, and turning curvature.
    print("Dubins path planner sample start!!")

    start_x = 1.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.deg2rad(45.0)  # [rad]

    end_x = -3.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.deg2rad(-45.0)  # [rad]

    curvature = 1.0

    # Generate the complete path in the global coordinate frame.
    path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature)

    # Plot the selected path and both endpoint poses when animation is enabled.
    show_animation = True
    if show_animation:
        plt.plot(path_x, path_y, label="".join(mode))
        plot_arrow(start_x, start_y, start_yaw)
        plot_arrow(end_x, end_y, end_yaw)
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    main()
