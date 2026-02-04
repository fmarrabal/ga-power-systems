# GA-Power-Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Geometric Algebra Power Theory (GAPoT) Framework for Power Systems Analysis**

This repository provides Python implementations of Geometric Algebra methods for electrical power systems analysis, accompanying the review article:

> Montoya, F.G., Alcayde, A., & Arrabal-Campos, F.M. (2026). *Geometric Algebra in Electrical Power Engineering: Foundations, Methods, and Applications*. Philosophical Transactions of the Royal Society A.

## 🎯 Features

- **GAPoT Framework**: Complete implementation of Geometric Algebra Power Theory
- **Fourier-to-GA Mapping**: Algorithm for transforming time-domain signals to geometric vectors
- **Power Decomposition**: Active, reactive, and distortion power calculation
- **Current Compensation**: Geometric projection/rejection for optimal compensation
- **Transformations**: Clarke, Park, and Fortescue transforms using rotors
- **Noise Mitigation**: Holoborodko, Tikhonov, and Total Variation regularization
- **Comparative Analysis**: Side-by-side comparison with IEEE 1459 and p-q theory

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/fmarrabal/ga-power-systems.git
cd ga-power-systems

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🚀 Quick Start

```python
import numpy as np
from gapot import GeometricPower, FourierToGA

# Define voltage and current signals (50 Hz fundamental + 5th harmonic)
t = np.linspace(0, 0.04, 2000)  # 2 cycles
omega1 = 2 * np.pi * 50

# Voltage: 230V fundamental + 5% 5th harmonic
u = 230*np.sqrt(2) * (np.sin(omega1*t) + 0.05*np.sin(5*omega1*t))

# Current: 10A fundamental (lagging 30°) + 20% 5th harmonic
i = 10*np.sqrt(2) * (np.sin(omega1*t - np.pi/6) + 0.20*np.sin(5*omega1*t + 0.5))

# Calculate geometric power
gp = GeometricPower(u, i, f1=50, fs=50000)

print(f"Active Power P = {gp.P:.2f} W")
print(f"Reactive Power ||M_Q|| = {gp.M_Q_norm:.2f} var")
print(f"Distortion Power ||M_D|| = {gp.M_D_norm:.2f} va")
print(f"Apparent Power ||M|| = {gp.S:.2f} VA")
print(f"Geometric Power Factor = {gp.PF:.4f}")
```

## 📁 Repository Structure

```
ga-power-systems/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
│
├── src/gapot/                   # Main package
│   ├── __init__.py              # Package exports
│   ├── basis.py                 # Fourier→GA mapping (Algorithm 1)
│   ├── clifford_ops.py          # Geometric algebra operations
│   ├── power.py                 # Geometric power calculation
│   ├── compensation.py          # Current decomposition
│   ├── transforms.py            # Clarke, Park, Fortescue
│   ├── noise.py                 # Noise mitigation techniques
│   └── traditional.py           # IEEE 1459, p-q for comparison
│
├── examples/                    # Jupyter notebooks
│   ├── 01_basic_power_calculation.ipynb
│   ├── 02_ieee1459_comparison.ipynb
│   ├── 03_pq_theory_comparison.ipynb
│   ├── 04_harmonic_analysis.ipynb
│   ├── 05_interharmonics.ipynb
│   ├── 06_three_phase_systems.ipynb
│   ├── 07_compensation_design.ipynb
│   └── 08_noise_sensitivity.ipynb
│
├── benchmarks/                  # Performance benchmarks
│   ├── computational_complexity.py
│   └── results/
│
├── data/                        # Test data
│   ├── experimental/            # From paper validation
│   └── synthetic/               # Generated test cases
│
├── tests/                       # Unit tests
│   ├── test_basis.py
│   ├── test_power.py
│   └── test_transforms.py
│
└── docs/                        # Documentation
    ├── theory_summary.md
    └── api_reference.md
```

## 📓 Examples

### Comparative Analysis: GAPoT vs IEEE 1459 vs p-q Theory

```python
from gapot import GeometricPower, IEEE1459Power, PQTheory
from gapot.utils import generate_distorted_signal

# Generate test signals
u, i, t = generate_distorted_signal(
    U1=230, I1=10, phi1=0.5,           # Fundamental
    harmonics=[(5, 0.05, 0.20, 0.3)],  # 5th harmonic
    f1=50, fs=50000, cycles=2
)

# Method 1: GAPoT
gapot = GeometricPower(u, i, f1=50, fs=50000)

# Method 2: IEEE 1459
ieee = IEEE1459Power(u, i, f1=50, fs=50000)

# Method 3: p-q Theory
pq = PQTheory(u, i, fs=50000)

# Compare results
print("=" * 60)
print(f"{'Method':<15} {'P (W)':<12} {'Q (var)':<12} {'D (va)':<12}")
print("=" * 60)
print(f"{'GAPoT':<15} {gapot.P:<12.2f} {gapot.M_Q_norm:<12.2f} {gapot.M_D_norm:<12.2f}")
print(f"{'IEEE 1459':<15} {ieee.P:<12.2f} {ieee.Q1:<12.2f} {ieee.D:<12.2f}")
print(f"{'p-q Theory':<15} {pq.p_avg:<12.2f} {pq.q_avg:<12.2f} {'N/A':<12}")
```

### Rotor-Based Transformations

```python
from gapot.transforms import ClarkeTransform, ParkTransform
import numpy as np

# Three-phase voltages
t = np.linspace(0, 0.04, 2000)
omega = 2 * np.pi * 50
va = 230*np.sqrt(2) * np.sin(omega*t)
vb = 230*np.sqrt(2) * np.sin(omega*t - 2*np.pi/3)
vc = 230*np.sqrt(2) * np.sin(omega*t + 2*np.pi/3)

# Clarke transform (abc → αβ)
clarke = ClarkeTransform()
v_alpha, v_beta = clarke.forward(va, vb, vc)

# Park transform (αβ → dq)
park = ParkTransform(omega=omega)
v_d, v_q = park.forward(v_alpha, v_beta, t)

# Extract rotor angle
theta = park.get_rotor_angle(t)
```

## 📊 Benchmarks

Run computational complexity benchmarks:

```bash
python benchmarks/computational_complexity.py
```

Results are saved to `benchmarks/results/` and include:
- Execution time vs. number of harmonics
- Memory usage analysis
- Comparison with complex algebra implementations

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=gapot --cov-report=html
```

## 📚 Documentation

- [Theory Summary](docs/theory_summary.md): Brief overview of GAPoT
- [API Reference](docs/api_reference.md): Complete function documentation
- [Examples](examples/): Jupyter notebooks with detailed explanations

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{montoya2026geometric,
  author  = {Montoya, Francisco G. and Alcayde, Alfredo and 
             Arrabal-Campos, Francisco M.},
  title   = {Geometric Algebra in Electrical Power Engineering: 
             Foundations, Methods, and Applications},
  journal = {Philosophical Transactions of the Royal Society A},
  year    = {2026},
  volume  = {XXX},
  pages   = {XXX--XXX},
  doi     = {10.1098/rsta.2025.XXXX}
}

@software{gapot2026,
  author    = {Montoya, Francisco G. and Alcayde, Alfredo and 
               Arrabal-Campos, Francisco M.},
  title     = {{GA-Power-Systems}: Geometric Algebra Power Theory Framework},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/fmarrabal/ga-power-systems}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Francisco G. Montoya** - *University of Almería* - [ORCID](https://orcid.org/XXXX-XXXX-XXXX-XXXX)
- **Alfredo Alcayde** - *University of Almería*
- **Francisco M. Arrabal-Campos** - *University of Almería* - [ORCID](https://orcid.org/0000-0001-9990-0569)

## 🙏 Acknowledgments

- Research Group FQM-376, University of Almería
- The `clifford` Python library developers
- All contributors to the Geometric Algebra research community

## 📬 Contact

For questions or collaboration inquiries:
- Email: pagilm@ual.es
- Issues: [GitHub Issues](https://github.com/fmarrabal/ga-power-systems/issues)
