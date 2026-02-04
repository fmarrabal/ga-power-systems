# Theoretical Foundations of Geometric Algebra Power Theory (GAPoT)

## Fundamentos Teóricos de la Teoría de Potencia con Álgebra Geométrica

---

## 1. Introduction / Introducción

### English

Geometric Algebra (GA), also known as Clifford Algebra, provides a unified mathematical framework that extends vector algebra to include operations that naturally encode geometric relationships. In the context of electrical power systems, GA offers significant advantages over traditional complex algebra, particularly for systems with harmonic distortion, unbalanced loads, or multiple frequencies.

The key insight of GAPoT is that voltage and current can be represented as **vectors** in a high-dimensional Euclidean space, where each frequency component occupies orthogonal dimensions. The **geometric product** of these vectors naturally separates power into its physical components: active (scalar), reactive (same-frequency bivector), and distortion (cross-frequency bivector).

### Español

El Álgebra Geométrica (GA), también conocida como Álgebra de Clifford, proporciona un marco matemático unificado que extiende el álgebra vectorial para incluir operaciones que codifican naturalmente las relaciones geométricas. En el contexto de los sistemas eléctricos de potencia, GA ofrece ventajas significativas sobre el álgebra compleja tradicional, particularmente para sistemas con distorsión armónica, cargas desequilibradas o múltiples frecuencias.

La idea clave de GAPoT es que tensión y corriente pueden representarse como **vectores** en un espacio Euclidiano de alta dimensión, donde cada componente frecuencial ocupa dimensiones ortogonales. El **producto geométrico** de estos vectores separa naturalmente la potencia en sus componentes físicos: activa (escalar), reactiva (bivector de misma frecuencia) y distorsión (bivector de frecuencia cruzada).

---

## 2. Mathematical Foundations / Fundamentos Matemáticos

### 2.1 The Geometric Product / El Producto Geométrico

For two vectors **a** and **b**, the geometric product is defined as:

$$\mathbf{ab} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \wedge \mathbf{b}$$

Where:
- $\mathbf{a} \cdot \mathbf{b}$ is the **inner product** (scalar): measures collinearity
- $\mathbf{a} \wedge \mathbf{b}$ is the **outer product** (bivector): represents the oriented plane containing both vectors

**Key properties:**
- The inner product is symmetric: $\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$
- The outer product is antisymmetric: $\mathbf{a} \wedge \mathbf{b} = -\mathbf{b} \wedge \mathbf{a}$
- Therefore, the geometric product is generally **non-commutative**: $\mathbf{ab} \neq \mathbf{ba}$

### 2.2 Basis Vectors / Vectores Base

For a system with N frequency components, we define an orthonormal basis:

$$\{\sigma_0, \sigma_1, \sigma_2, \sigma_3, \sigma_4, \ldots\}$$

Where:
- $\sigma_0$: DC component
- $\sigma_{2k-1}$: Cosine component at frequency $f_k$
- $\sigma_{2k}$: Sine component at frequency $f_k$

These basis vectors satisfy:
$$\sigma_i \cdot \sigma_j = \delta_{ij}$$

### 2.3 Signal Representation / Representación de Señales

A periodic voltage signal:

$$u(t) = U_0 + \sum_{h=1}^{N} \left[ U_{hc} \cos(h\omega_1 t) + U_{hs} \sin(h\omega_1 t) \right]$$

is represented as the GA vector:

$$\mathbf{u} = U_0 \sigma_0 + \sum_{h=1}^{N} \left[ U_{hc} \sigma_{2h-1} + U_{hs} \sigma_{2h} \right]$$

Similarly for current:

$$\mathbf{i} = I_0 \sigma_0 + \sum_{h=1}^{N} \left[ I_{hc} \sigma_{2h-1} + I_{hs} \sigma_{2h} \right]$$

---

## 3. Geometric Power / Potencia Geométrica

### 3.1 Definition / Definición

The **geometric power** is defined as the geometric product of voltage and current vectors:

$$\mathbf{M} = \mathbf{u}\mathbf{i} = P + \mathbf{M}_Q + \mathbf{M}_D$$

### 3.2 Active Power (Scalar Part) / Potencia Activa (Parte Escalar)

$$P = \mathbf{u} \cdot \mathbf{i} = U_0 I_0 + \sum_{h=1}^{N} \frac{1}{2}(U_{hc}I_{hc} + U_{hs}I_{hs})$$

This represents the **average power transferred** from source to load.

For a single harmonic h:
$$P_h = \frac{1}{2}(U_{hc}I_{hc} + U_{hs}I_{hs}) = U_h I_h \cos(\phi_h)$$

where $\phi_h$ is the phase angle between voltage and current at harmonic h.

### 3.3 Reactive Power (Same-Frequency Bivector) / Potencia Reactiva (Bivector de Misma Frecuencia)

$$\mathbf{M}_Q = \sum_{h=1}^{N} Q_h \mathbf{B}_h$$

where:
$$Q_h = \frac{1}{2}(U_{hc}I_{hs} - U_{hs}I_{hc}) = U_h I_h \sin(\phi_h)$$

and $\mathbf{B}_h = \sigma_{2h-1} \wedge \sigma_{2h}$ is the bivector representing the plane of oscillation at frequency h.

This represents **energy oscillation** between source and reactive elements (inductors, capacitors) at each frequency.

### 3.4 Distortion Power (Cross-Frequency Bivector) / Potencia de Distorsión (Bivector de Frecuencia Cruzada)

$$\mathbf{M}_D = \sum_{m \neq n} D_{mn} \mathbf{B}_{mn}$$

where:
$$D_{mn} = \frac{1}{2}\left[(U_{mc}I_{ns} - U_{ms}I_{nc}) - (U_{nc}I_{ms} - U_{ns}I_{mc})\right]$$

and $\mathbf{B}_{mn} = \sigma_m \wedge \sigma_n$ represents the interaction between different frequencies.

This component **has no physical interpretation in terms of energy flow** but represents the mathematical consequence of multiplying signals at different frequencies.

### 3.5 Apparent Power and Power Factor / Potencia Aparente y Factor de Potencia

The apparent power is the norm of the geometric power:

$$S = \|\mathbf{M}\| = \sqrt{P^2 + \|\mathbf{M}_Q\|^2 + \|\mathbf{M}_D\|^2}$$

The **geometric power factor** is:

$$PF_g = \frac{P}{\|\mathbf{M}\|} = \frac{P}{S}$$

### 3.6 Energy Conservation / Conservación de Energía

A fundamental property of GAPoT is that it satisfies energy conservation:

$$\|\mathbf{M}\|^2 = P^2 + \|\mathbf{M}_Q\|^2 + \|\mathbf{M}_D\|^2 = U_{rms}^2 \times I_{rms}^2 = S^2$$

This is consistent with Parseval's theorem and the traditional definition of apparent power.

---

## 4. Current Decomposition / Descomposición de Corriente

### 4.1 Geometric Projection / Proyección Geométrica

The current vector can be decomposed using geometric projection and rejection:

**Active current** (parallel to voltage):
$$\mathbf{i}_P = \frac{\mathbf{i} \cdot \mathbf{u}}{\|\mathbf{u}\|^2} \mathbf{u} = G_e \mathbf{u}$$

where $G_e = P / \|\mathbf{u}\|^2$ is the equivalent conductance.

**Non-active current** (orthogonal to voltage):
$$\mathbf{i}_N = \mathbf{i} - \mathbf{i}_P$$

### 4.2 Compensation Strategy / Estrategia de Compensación

For unity power factor, the compensation current must cancel the non-active component:

$$\mathbf{i}_{comp} = -\mathbf{i}_N = -(\mathbf{i} - \mathbf{i}_P)$$

After compensation, the supply current equals the active current:
$$\mathbf{i}_{supply} = \mathbf{i} + \mathbf{i}_{comp} = \mathbf{i}_P$$

---

## 5. Rotors and Transformations / Rotores y Transformaciones

### 5.1 Rotor Definition / Definición de Rotor

A **rotor** represents a rotation in GA:

$$R = \cos(\theta/2) - \mathbf{B}\sin(\theta/2) = e^{-\mathbf{B}\theta/2}$$

where $\mathbf{B}$ is a unit bivector defining the rotation plane and $\theta$ is the rotation angle.

To rotate a vector **v**:
$$\mathbf{v}' = R\mathbf{v}\tilde{R}$$

where $\tilde{R} = \cos(\theta/2) + \mathbf{B}\sin(\theta/2)$ is the reverse of R.

### 5.2 Advantages over Matrices / Ventajas sobre Matrices

| Property | Matrix | Rotor |
|----------|--------|-------|
| Rotation plane | Implicit in entries | Explicit bivector **B** |
| Angle extraction | arctan(m₂₁/m₁₁) | Direct: 2·arctan(|⟨R⟩₂|/⟨R⟩₀) |
| Composition | Matrix multiplication | Geometric product R₂R₁ |
| Inverse | Matrix inversion | Simple reverse: R̃ |
| Interpolation | Requires decomposition | Natural SLERP |
| Singularities | Gimbal lock possible | None (double cover) |

### 5.3 Park Transform Example / Ejemplo de Transformada de Park

The Park transform rotates from stationary αβ to rotating dq frame:

**Matrix form:**
$$\begin{bmatrix} v_d \\ v_q \end{bmatrix} = \begin{bmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} v_\alpha \\ v_\beta \end{bmatrix}$$

**Rotor form:**
$$R(t) = e^{-\mathbf{B}_{12}\theta(t)/2}$$

where $\mathbf{B}_{12} = \sigma_\alpha \wedge \sigma_\beta$ and $\theta(t) = \omega t$.

---

## 6. Comparison with Traditional Methods / Comparación con Métodos Tradicionales

### 6.1 IEEE 1459

IEEE 1459 defines:
- $S^2 = P^2 + Q_1^2 + D^2$
- Distortion D aggregates all non-fundamental contributions

**Limitation:** Cannot distinguish individual cross-frequency interactions.

### 6.2 p-q Theory (Akagi)

Instantaneous power theory:
- $p = u_\alpha i_\alpha + u_\beta i_\beta$
- $q = u_\beta i_\alpha - u_\alpha i_\beta$

**Limitation:** Primarily designed for fundamental frequency; requires extensions for harmonics.

### 6.3 GAPoT Advantages

1. **Explicit cross-frequency terms:** Each U_m × I_n interaction has distinct representation
2. **Geometric interpretation:** Bivectors represent oriented planes
3. **Energy conservation:** Norm equals traditional apparent power
4. **Unified framework:** Single algebra for transformations, power, and compensation

---

## 7. Practical Considerations / Consideraciones Prácticas

### 7.1 Computational Complexity

For N harmonics, GAPoT operates in dimension 2N+1 (including DC).

- Vector operations: O(N)
- Geometric product: O(N²) for bivector terms
- Same complexity as IEEE 1459 harmonic analysis

### 7.2 Noise Sensitivity

Derivative-based methods amplify high-frequency noise. Recommended mitigations:
- Holoborodko smooth differentiators
- Tikhonov (L2) regularization
- Total Variation (L1) regularization

### 7.3 Implementation Notes

1. Use FFT for Fourier-to-GA transformation (efficient O(N log N))
2. Unified frequency basis ensures voltage and current are in same space
3. Energy conservation check validates implementation correctness

---

## References / Referencias

1. Menti, A., Zacharias, T., & Milias-Argitis, J. (2007). Geometric algebra: A powerful tool for representing power under nonsinusoidal conditions. *IEEE Trans. Circuits Syst. I*, 54(3), 601-609.

2. Castro-Núñez, M., & Castro-Puche, R. (2012). The IEEE Standard 1459, the CPC Power Theory, and Geometric Algebra in Circuits with Nonsinusoidal Sources and Linear Loads. *IEEE Trans. Circuits Syst. I*, 59(12), 2980-2990.

3. Montoya, F.G., Baños, R., Alcayde, A., & Arrabal-Campos, F.M. (2021). Vector Geometric Algebra in Power Systems: An Updated Formulation of Apparent Power under Non-Sinusoidal Conditions. *Mathematics*, 9(11), 1295.

4. Hestenes, D., & Sobczyk, G. (1984). *Clifford Algebra to Geometric Calculus: A Unified Language for Mathematics and Physics*. Springer.

5. Chappell, J.M., Drake, S.P., Seidel, C.L., Gunn, L.J., Iqbal, A., & Abbott, D. (2014). Geometric Algebra for Electrical and Electronic Engineers. *Proc. IEEE*, 102(9), 1340-1363.
