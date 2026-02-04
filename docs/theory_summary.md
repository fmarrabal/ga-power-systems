# Geometric Algebra Power Theory: Theoretical Summary

## 1. Introduction

Geometric Algebra (GA), also known as Clifford Algebra, provides a unified mathematical framework for representing and manipulating geometric objects. In the context of electrical power systems, GA offers a powerful alternative to traditional complex algebra, particularly for systems with harmonic distortion.

## 2. Fundamental Concepts

### 2.1 Geometric Product

The geometric product of two vectors **a** and **b** combines the inner (dot) and outer (wedge) products:

**ab** = **a** · **b** + **a** ∧ **b**

- **a** · **b** (scalar): Represents collinear components
- **a** ∧ **b** (bivector): Represents perpendicular components, encoding the plane containing both vectors

### 2.2 Multivectors

In GA, quantities can be multivectors containing different grades:
- Grade 0: Scalars (numbers)
- Grade 1: Vectors
- Grade 2: Bivectors (oriented planes)
- Grade n: n-vectors

### 2.3 Basis Vectors

For power systems with N frequency components:
- **σ₀**: DC component
- **σ₂ₘ₋₁**, **σ₂ₘ**: Cosine and sine components at frequency fₘ

## 3. Fourier-to-GA Mapping

### 3.1 Signal Representation

A periodic voltage signal with Fourier decomposition:

u(t) = U₀ + Σ [Uₕc cos(hω₁t) + Uₕs sin(hω₁t)]

is represented as the GA vector:

**u** = U₀**σ₀** + Σ [Uₕc**σ₂ₕ₋₁** + Uₕs**σ₂ₕ**]

### 3.2 Indexing Strategy

| Component | Frequency | Cosine Basis | Sine Basis |
|-----------|-----------|--------------|------------|
| DC | 0 | **σ₀** | — |
| Fundamental | f₁ | **σ₁** | **σ₂** |
| 2nd Harmonic | 2f₁ | **σ₃** | **σ₄** |
| h-th Harmonic | hf₁ | **σ₂ₕ₋₁** | **σ₂ₕ** |

## 4. Geometric Power

### 4.1 Definition

The geometric power is defined as the geometric product of voltage and current vectors:

**M** = **u** **i** = P + **M**_Q + **M**_D

### 4.2 Power Components

**Active Power (Scalar)**:
P = Σₕ ½(Uₕc Iₕc + Uₕs Iₕs)

**Reactive Power (Same-frequency Bivector)**:
**M**_Q = Σₕ ½(Uₕc Iₕs - Uₕs Iₕc) **Bₕ**

**Distortion Power (Cross-frequency Bivector)**:
**M**_D = Σₘ≠ₙ (Uₘc Iₙs - Uₘs Iₙc) **σₘ** ∧ **σₙ**

### 4.3 Energy Conservation

The geometric power norm satisfies:

‖**M**‖² = P² + ‖**M**_Q‖² + ‖**M**_D‖² = U²_rms × I²_rms = S²

This is consistent with Parseval's theorem and traditional apparent power definition.

## 5. Current Decomposition

### 5.1 Projection and Rejection

Using GA operations, current can be decomposed:

**i**_P = (**i** · **u** / ‖**u**‖²) **u**  (Active current)

**i**_N = **i** - **i**_P  (Non-active current)

### 5.2 Compensation Strategy

The compensation current is:

**i**_comp = -**i**_N

which, when injected, leaves only the active current component, achieving unity power factor.

## 6. Transformations with Rotors

### 6.1 Rotor Definition

A rotor R = cos(θ/2) - **B** sin(θ/2) represents rotation by angle θ in the plane defined by bivector **B**.

### 6.2 Advantages over Matrices

| Property | Matrix | Rotor |
|----------|--------|-------|
| Rotation plane | Implicit | Explicit **B** |
| Angle extraction | arctan(m₂₁/m₁₁) | Direct from components |
| Composition | Matrix multiply | Geometric product |
| Inverse | Matrix inversion | Reverse: R̃ |

## 7. Comparison with Traditional Methods

### 7.1 IEEE 1459

IEEE 1459 defines:
- S² = P² + Q₁² + D²
- D = √(S² - P² - Q₁²)

The distortion D aggregates all non-fundamental effects without distinguishing cross-frequency interactions.

### 7.2 p-q Theory

Instantaneous power theory provides:
- p = u_α i_α + u_β i_β
- q = u_β i_α - u_α i_β

Limited to fundamental frequency analysis in standard form.

### 7.3 GA Advantages

1. **Explicit cross-frequency terms**: Each interaction U_m × I_n has a distinct representation
2. **Geometric interpretation**: Bivectors represent oriented planes
3. **Energy conservation**: Norm equals traditional apparent power
4. **Unified framework**: Single algebra for all analyses

## 8. References

1. Menti, A. et al. (2007). Geometric algebra: A powerful tool for representing power under nonsinusoidal conditions. *IEEE Trans. Circuits Syst. I*.

2. Castro-Núñez, M. & Castro-Puche, R. (2012). The IEEE Standard 1459, the CPC Power Theory, and Geometric Algebra. *IEEE Trans. Circuits Syst. I*.

3. Montoya, F.G. et al. (2021). Vector Geometric Algebra in Power Systems: An Updated Formulation of Apparent Power. *Mathematics*.

4. Chappell, J.M. et al. (2014). Geometric Algebra for Electrical and Electronic Engineers. *Proc. IEEE*.
