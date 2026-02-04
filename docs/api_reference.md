# API Reference

Complete documentation for the GAPoT (Geometric Algebra Power Theory) framework.

## Core Classes

### `GeometricPower`

Main class for calculating geometric power from voltage and current signals.

```python
from gapot import GeometricPower

gp = GeometricPower(u, i, f1=50, fs=10000)
```

**Parameters:**
- `u` (np.ndarray): Voltage signal samples
- `i` (np.ndarray): Current signal samples  
- `f1` (float): Fundamental frequency in Hz
- `fs` (float): Sampling frequency in Hz
- `threshold` (float, optional): FFT magnitude threshold. Default 0.01

**Attributes:**
- `P` (float): Active power in Watts
- `M_Q_norm` (float): Reactive power magnitude in var
- `M_D_norm` (float): Distortion power magnitude
- `S` (float): Apparent power in VA
- `PF` (float): Geometric power factor
- `u_vec` (np.ndarray): Voltage GA vector
- `i_vec` (np.ndarray): Current GA vector

**Methods:**
- `get_power_components()`: Returns `PowerComponents` dataclass
- `get_harmonic_powers()`: Returns dict mapping frequency to {P, Q} values
- `verify_energy_conservation()`: Verifies ||M||² = U_rms² × I_rms²
- `summary()`: Returns formatted text summary

---

### `FourierToGA`

Transforms time-domain signals to geometric algebra vector representation.

```python
from gapot import FourierToGA

converter = FourierToGA(fs=10000, f1=50)
ga_vector, components = converter.transform(signal)
```

**Parameters:**
- `fs` (float): Sampling frequency in Hz
- `f1` (float): Fundamental frequency in Hz
- `threshold` (float, optional): Magnitude threshold. Default 0.01
- `max_harmonics` (int, optional): Maximum harmonics to consider. Default 50

**Methods:**
- `transform(signal)`: Returns (ga_vector, components) tuple
- `inverse_transform(ga_vector, t)`: Reconstructs time-domain signal
- `get_frequency_mapping()`: Returns dict mapping frequency to basis indices
- `verify_parseval(signal, ga_vector)`: Verifies energy conservation

---

### `CurrentDecomposition`

Decomposes current into active, reactive, and distortion components.

```python
from gapot import CurrentDecomposition

decomp = CurrentDecomposition(u_vec, i_vec)
i_active = decomp.i_active
i_reactive = decomp.i_reactive
```

**Parameters:**
- `u_vec` (np.ndarray): Voltage GA vector
- `i_vec` (np.ndarray): Current GA vector
- `frequency_info` (dict, optional): Frequency mapping from FourierToGA

**Attributes:**
- `i_active` (np.ndarray): Active current vector
- `i_reactive` (np.ndarray): Reactive current vector
- `i_distortion` (np.ndarray): Distortion current vector
- `i_active_norm`, `i_reactive_norm`, `i_distortion_norm` (float): Norms

**Methods:**
- `get_components()`: Returns `CurrentComponents` dataclass
- `get_compensation_current()`: Returns required compensation current
- `verify_orthogonality()`: Verifies orthogonality of components

---

## Transformation Classes

### `ClarkeTransform`

Clarke (αβ) transformation using geometric interpretation.

```python
from gapot import ClarkeTransform

clarke = ClarkeTransform(amplitude_invariant=True)
v_alpha, v_beta = clarke.forward(v_a, v_b, v_c)
v_a, v_b, v_c = clarke.inverse(v_alpha, v_beta)
```

### `ParkTransform`

Park (dq) transformation using GA rotors.

```python
from gapot import ParkTransform

park = ParkTransform(omega=2*np.pi*50, theta0=0)
v_d, v_q = park.forward(v_alpha, v_beta, t)
v_alpha, v_beta = park.inverse(v_d, v_q, t)
```

### `FortescueTransform`

Symmetrical components (positive, negative, zero sequence).

```python
from gapot import FortescueTransform

fortescue = FortescueTransform()
v_pos, v_neg, v_zero = fortescue.forward(v_a, v_b, v_c)
```

---

## Traditional Methods (for comparison)

### `IEEE1459Power`

IEEE Standard 1459-2010 power calculations.

```python
from gapot import IEEE1459Power

ieee = IEEE1459Power(u, i, f1=50, fs=10000)
print(f"P1={ieee.P1}, Q1={ieee.Q1}, D={ieee.D}")
```

### `PQTheory`

Instantaneous power (p-q) theory by Akagi.

```python
from gapot import PQTheory

pq = PQTheory(u, i, fs=10000)
print(f"p_avg={pq.p_avg}, q_avg={pq.q_avg}")
```

---

## Noise Mitigation

### `HoloborodkoFilter`

Smooth noise-robust differentiator.

```python
from gapot import HoloborodkoFilter

filt = HoloborodkoFilter(order=7, dt=1e-4)
dx = filt.differentiate(x)
```

### `TikhonovRegularizer`

L2 regularization for noisy differentiation.

```python
from gapot import TikhonovRegularizer

reg = TikhonovRegularizer(lambda_reg=1e-3)
dx = reg.differentiate(x, dt=1e-4)
```

### `TVRegularizer`

Total Variation (L1) regularization.

```python
from gapot import TVRegularizer

reg = TVRegularizer(lambda_reg=1e-2)
dx = reg.differentiate(x, dt=1e-4)
```

---

## Utility Functions

### `generate_distorted_signal`

Generate test signals with harmonics.

```python
from gapot.utils import generate_distorted_signal

u, i, t = generate_distorted_signal(
    U1=230, I1=10, phi1=0.5,
    harmonics=[(5, 0.05, 0.20, 0.3)],
    f1=50, fs=10000, cycles=2
)
```

### `rms`, `thd`

Calculate RMS value and Total Harmonic Distortion.

```python
from gapot.utils import rms, thd

voltage_rms = rms(u)
current_thd = thd(i, f1=50, fs=10000)
```

---

## Data Classes

### `PowerComponents`
```python
@dataclass
class PowerComponents:
    P: float           # Active power
    M_Q: np.ndarray    # Reactive power bivector coefficients
    M_D: np.ndarray    # Distortion power bivector coefficients
    M_Q_norm: float    # ||M_Q||
    M_D_norm: float    # ||M_D||
    S: float           # Apparent power
    PF: float          # Power factor
```

### `SpectralComponent`
```python
@dataclass
class SpectralComponent:
    frequency: float
    magnitude: float
    phase: float
    cosine_coeff: float
    sine_coeff: float
    basis_index_cos: int
    basis_index_sin: Optional[int]
    is_dc: bool
```
