"""
Electrical Transformations using Geometric Algebra
===================================================

Implementation of Clarke, Park, and Fortescue transformations
using GA rotors instead of traditional matrix representations.

The rotor representation provides:
- Explicit rotation plane identification (bivector)
- Direct angle extraction
- Simplified composition and inversion
- No gimbal lock or singularities

References
----------
.. [1] Chappell et al. (2014). Geometric Algebra for Electrical 
       and Electronic Engineers.
.. [2] Petroianu (2015). A geometric algebra reformulation of 
       Steinmetz's symbolic method.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RotorParams:
    """Parameters describing a rotor transformation."""
    angle: float           # Rotation angle in radians
    plane_indices: Tuple[int, int]  # Indices of basis vectors defining plane
    scalar_part: float     # cos(θ/2)
    bivector_part: float   # sin(θ/2)


class Rotor:
    """
    Geometric Algebra Rotor for rotations.
    
    A rotor R = cos(θ/2) - B sin(θ/2) represents a rotation by angle θ
    in the plane defined by bivector B.
    
    The rotation of a vector v is: v' = R v R†
    where R† is the reverse (conjugate) of R.
    
    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    plane : tuple of int
        Indices (i, j) of basis vectors defining rotation plane.
        The bivector B = e_i ∧ e_j.
        
    Example
    -------
    >>> rotor = Rotor(angle=np.pi/4, plane=(0, 1))  # 45° in xy-plane
    >>> v = np.array([1.0, 0.0, 0.0])
    >>> v_rotated = rotor.apply(v)
    """
    
    def __init__(self, angle: float, plane: Tuple[int, int] = (0, 1)):
        self.angle = angle
        self.plane = plane
        
        # Rotor components: R = a + b*B where B is the unit bivector
        self.a = np.cos(angle / 2)  # Scalar part
        self.b = np.sin(angle / 2)  # Bivector coefficient
    
    @property
    def reverse(self) -> 'Rotor':
        """Return the reverse (conjugate) R† = a - b*B."""
        rev = Rotor(0, self.plane)
        rev.a = self.a
        rev.b = -self.b
        rev.angle = -self.angle
        return rev
    
    def apply(self, v: np.ndarray) -> np.ndarray:
        """
        Apply rotation to vector: v' = R v R†.
        
        For a 2D rotation in the plane (i, j):
        v'_i = v_i cos(θ) - v_j sin(θ)
        v'_j = v_i sin(θ) + v_j cos(θ)
        """
        v_out = v.copy()
        i, j = self.plane
        
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        
        v_out[i] = v[i] * c - v[j] * s
        v_out[j] = v[i] * s + v[j] * c
        
        return v_out
    
    def compose(self, other: 'Rotor') -> 'Rotor':
        """Compose two rotors: R_12 = R_2 * R_1 (R_1 applied first)."""
        if self.plane != other.plane:
            raise NotImplementedError("Composition of rotors in different planes")
        
        # For same-plane rotors, angles simply add
        return Rotor(self.angle + other.angle, self.plane)
    
    def get_params(self) -> RotorParams:
        """Get rotor parameters."""
        return RotorParams(
            angle=self.angle,
            plane_indices=self.plane,
            scalar_part=self.a,
            bivector_part=self.b
        )
    
    def __repr__(self) -> str:
        return f"Rotor(angle={self.angle:.4f} rad, plane={self.plane})"


class ClarkeTransform:
    """
    Clarke (αβ) Transformation using GA.
    
    Transforms three-phase abc quantities to two-phase αβ quantities.
    
    Traditional matrix form:
    [v_α]   [1   -1/2   -1/2 ] [v_a]
    [v_β] = [0  √3/2  -√3/2 ] [v_b]
                              [v_c]
    
    GA interpretation: Projection onto the αβ plane orthogonal to the
    (1,1,1) direction (zero-sequence).
    
    Parameters
    ----------
    amplitude_invariant : bool
        If True, use amplitude-invariant (power-invariant) form with
        factor √(2/3). If False, use standard form with factor 2/3.
        
    Example
    -------
    >>> clarke = ClarkeTransform()
    >>> v_a, v_b, v_c = 230*np.sin(wt), 230*np.sin(wt-2*pi/3), 230*np.sin(wt+2*pi/3)
    >>> v_alpha, v_beta = clarke.forward(v_a, v_b, v_c)
    """
    
    def __init__(self, amplitude_invariant: bool = False):
        self.amplitude_invariant = amplitude_invariant
        
        if amplitude_invariant:
            self.k = np.sqrt(2/3)
        else:
            self.k = 2/3
        
        # Transformation matrix
        self.T = self.k * np.array([
            [1, -0.5, -0.5],
            [0, np.sqrt(3)/2, -np.sqrt(3)/2]
        ])
        
        # Inverse transformation
        if amplitude_invariant:
            self.T_inv = np.sqrt(2/3) * np.array([
                [1, 0],
                [-0.5, np.sqrt(3)/2],
                [-0.5, -np.sqrt(3)/2]
            ])
        else:
            self.T_inv = np.array([
                [1, 0],
                [-0.5, np.sqrt(3)/2],
                [-0.5, -np.sqrt(3)/2]
            ])
    
    def forward(self, v_a: np.ndarray, v_b: np.ndarray, 
                v_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform abc to αβ.
        
        Returns
        -------
        v_alpha, v_beta : np.ndarray
            Alpha and beta components.
        """
        v_abc = np.vstack([v_a, v_b, v_c])
        v_alphabeta = self.T @ v_abc
        return v_alphabeta[0], v_alphabeta[1]
    
    def inverse(self, v_alpha: np.ndarray, 
                v_beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform αβ to abc.
        
        Returns
        -------
        v_a, v_b, v_c : np.ndarray
            Three-phase components.
        """
        v_alphabeta = np.vstack([v_alpha, v_beta])
        v_abc = self.T_inv @ v_alphabeta
        return v_abc[0], v_abc[1], v_abc[2]
    
    def get_projection_bivector(self) -> str:
        """
        Get the bivector representation of the projection plane.
        
        The αβ plane is orthogonal to the zero-sequence direction (1,1,1).
        """
        return "B_αβ = σ_1 ∧ σ_2 (in the space orthogonal to (1,1,1))"


class ParkTransform:
    """
    Park (dq) Transformation using GA rotors.
    
    Transforms αβ quantities to rotating dq frame.
    
    Traditional form:
    [v_d]   [cos(θ)   sin(θ)] [v_α]
    [v_q] = [-sin(θ)  cos(θ)] [v_β]
    
    GA interpretation: Rotation by -θ in the αβ plane using rotor
    R(t) = exp(-B_αβ θ(t)/2)
    
    Parameters
    ----------
    omega : float
        Angular frequency of rotation (rad/s).
    theta0 : float
        Initial angle (rad). Default 0.
        
    Example
    -------
    >>> park = ParkTransform(omega=2*np.pi*50)
    >>> v_d, v_q = park.forward(v_alpha, v_beta, t)
    """
    
    def __init__(self, omega: float, theta0: float = 0.0):
        self.omega = omega
        self.theta0 = theta0
    
    def get_angle(self, t: np.ndarray) -> np.ndarray:
        """Get rotation angle at time t."""
        return self.omega * t + self.theta0
    
    def get_rotor(self, t: float) -> Rotor:
        """Get rotor at time t."""
        theta = self.omega * t + self.theta0
        return Rotor(angle=-theta, plane=(0, 1))
    
    def get_rotor_angle(self, t: np.ndarray) -> np.ndarray:
        """Get rotor angle for array of time points."""
        return self.get_angle(t)
    
    def forward(self, v_alpha: np.ndarray, v_beta: np.ndarray,
                t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform αβ to dq.
        
        Returns
        -------
        v_d, v_q : np.ndarray
            Direct and quadrature components.
        """
        theta = self.get_angle(t)
        c = np.cos(theta)
        s = np.sin(theta)
        
        v_d = v_alpha * c + v_beta * s
        v_q = -v_alpha * s + v_beta * c
        
        return v_d, v_q
    
    def inverse(self, v_d: np.ndarray, v_q: np.ndarray,
                t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform dq to αβ.
        
        Returns
        -------
        v_alpha, v_beta : np.ndarray
            Alpha and beta components.
        """
        theta = self.get_angle(t)
        c = np.cos(theta)
        s = np.sin(theta)
        
        v_alpha = v_d * c - v_q * s
        v_beta = v_d * s + v_q * c
        
        return v_alpha, v_beta
    
    def get_rotation_bivector(self) -> str:
        """Get the bivector defining the rotation plane."""
        return "B_12 = σ_α ∧ σ_β"


class FortescueTransform:
    """
    Fortescue (Symmetrical Components) Transformation using GA.
    
    Transforms abc quantities to positive, negative, and zero sequence.
    
    Traditional form uses complex operator a = exp(j2π/3).
    
    GA interpretation: Rotors in the αβ plane with angles ±2π/3.
    
    Example
    -------
    >>> fortescue = FortescueTransform()
    >>> v_pos, v_neg, v_zero = fortescue.forward(v_a, v_b, v_c)
    """
    
    def __init__(self):
        # Complex operator a = exp(j2π/3)
        self.a = np.exp(1j * 2 * np.pi / 3)
        self.a2 = self.a ** 2
        
        # Transformation matrix
        self.T = (1/3) * np.array([
            [1, self.a, self.a2],
            [1, self.a2, self.a],
            [1, 1, 1]
        ])
        
        # Inverse
        self.T_inv = np.array([
            [1, 1, 1],
            [self.a2, self.a, 1],
            [self.a, self.a2, 1]
        ])
        
        # GA rotors for positive and negative sequence
        self.rotor_pos = Rotor(angle=2*np.pi/3, plane=(0, 1))
        self.rotor_neg = Rotor(angle=-2*np.pi/3, plane=(0, 1))
    
    def forward(self, v_a: np.ndarray, v_b: np.ndarray,
                v_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform abc to symmetrical components (complex).
        
        Returns
        -------
        v_pos, v_neg, v_zero : np.ndarray
            Positive, negative, and zero sequence components (complex).
        """
        # Convert to phasors (complex) if real
        if np.isrealobj(v_a):
            # Assume fundamental frequency, extract phasor
            # This is simplified - full implementation would use FFT
            v_a_c = v_a + 0j
            v_b_c = v_b + 0j
            v_c_c = v_c + 0j
        else:
            v_a_c, v_b_c, v_c_c = v_a, v_b, v_c
        
        v_abc = np.vstack([v_a_c, v_b_c, v_c_c])
        v_seq = self.T @ v_abc
        
        return v_seq[0], v_seq[1], v_seq[2]
    
    def inverse(self, v_pos: np.ndarray, v_neg: np.ndarray,
                v_zero: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transform symmetrical components to abc.
        
        Returns
        -------
        v_a, v_b, v_c : np.ndarray
            Three-phase components.
        """
        v_seq = np.vstack([v_pos, v_neg, v_zero])
        v_abc = self.T_inv @ v_seq
        
        return v_abc[0], v_abc[1], v_abc[2]
    
    def get_sequence_rotors(self) -> dict:
        """Get rotors for sequence component interpretation."""
        return {
            'positive': self.rotor_pos,
            'negative': self.rotor_neg,
            'zero': None  # Zero sequence has no rotation
        }


def rotation_matrix_from_rotor(rotor: Rotor, dimension: int = 3) -> np.ndarray:
    """
    Convert a rotor to equivalent rotation matrix.
    
    This demonstrates the equivalence between GA rotor and matrix
    representations.
    """
    R = np.eye(dimension)
    i, j = rotor.plane
    
    c = np.cos(rotor.angle)
    s = np.sin(rotor.angle)
    
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    
    return R


def rotor_from_rotation_matrix(R: np.ndarray, plane: Tuple[int, int] = (0, 1)) -> Rotor:
    """
    Extract rotor from rotation matrix.
    
    Parameters
    ----------
    R : np.ndarray
        Rotation matrix.
    plane : tuple
        Which plane to extract rotation from.
        
    Returns
    -------
    rotor : Rotor
        Equivalent rotor.
    """
    i, j = plane
    angle = np.arctan2(R[j, i], R[i, i])
    return Rotor(angle=angle, plane=plane)


def slerp(rotor1: Rotor, rotor2: Rotor, t: float) -> Rotor:
    """
    Spherical linear interpolation between two rotors.
    
    Parameters
    ----------
    rotor1, rotor2 : Rotor
        Start and end rotors (must be in same plane).
    t : float
        Interpolation parameter in [0, 1].
        
    Returns
    -------
    rotor : Rotor
        Interpolated rotor.
    """
    if rotor1.plane != rotor2.plane:
        raise ValueError("SLERP requires rotors in the same plane")
    
    # For same-plane rotors, SLERP is linear in angle
    angle = (1 - t) * rotor1.angle + t * rotor2.angle
    return Rotor(angle=angle, plane=rotor1.plane)
