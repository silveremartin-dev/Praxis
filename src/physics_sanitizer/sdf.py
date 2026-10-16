"""
Analytical Signed Distance Fields (SDF) for rigid geometric primitives.
Used by the physics sanitizer to enforce strictly zero mesh interpenetration
and compute exact contact distances without polygon ray casting.
"""

import numpy as np

def sdf_cylinder(points: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    """
    Computes exact Signed Distance Field for a cylinder centered at origin oriented along Z-axis.
    Returns:
        distances: (N,) array where:
          > 0 : strictly outside the cylinder
          = 0 : on the cylinder boundary surface
          < 0 : penetrating inside the cylinder
    """
    p = np.atleast_2d(points)
    radial_dist = np.linalg.norm(p[:, :2], axis=1) - radius
    axial_dist = np.abs(p[:, 2]) - half_height
    
    # Outside distance
    d_x = np.maximum(radial_dist, 0.0)
    d_y = np.maximum(axial_dist, 0.0)
    outside = np.hypot(d_x, d_y)
    
    # Inside distance (negative)
    inside = np.minimum(np.maximum(radial_dist, axial_dist), 0.0)
    return outside + inside

def sdf_box(points: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    """
    Computes exact Signed Distance Field for a box centered at origin.
    half_extents: [hx, hy, hz]
    """
    p = np.atleast_2d(points)
    he = np.array(half_extents)
    q = np.abs(p) - he
    
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside

def sdf_sphere(points: np.ndarray, radius: float) -> np.ndarray:
    """
    Computes exact Signed Distance Field for a sphere centered at origin.
    """
    p = np.atleast_2d(points)
    return np.linalg.norm(p, axis=1) - radius
