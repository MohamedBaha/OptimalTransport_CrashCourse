# Optimal Transport — Crash Course

A comprehensive beginner's course on Optimal Transport with accompanying exercises and solutions. This repository contains lecture materials, Jupyter notebooks with theory, and hands-on programming exercises to build intuition for optimal transport concepts.

## 📚 Course Overview

Optimal Transport (OT) is a powerful mathematical framework for comparing probability distributions. It has applications spanning:
- Machine learning (domain adaptation, generative models, WGAN)
- Computer vision (image processing, shape analysis)
- Economics (matching problems, market design)
- Scientific computing (PDE-constrained optimization)

This course progresses from foundational concepts to advanced applications, with a focus on intuitive understanding and practical implementation.

## 📖 Repository Structure

```
OptimalTransport_CrashCourse/
├── README.md                              # This file
├── optimal-transport-course.ipynb          # Main course material (8 modules)
├── optimal-transport-exercises.ipynb       # Companion exercises (12 problems)
└── [Additional materials]
```

## 🎯 Course Modules

### Module 0: Motivation
- Why Wasserstein distances matter
- Comparison with classical divergences (KL, TV, JS)
- Geometric intuition: why OT is geometry-aware

### Module 1: The Discrete Kantorovich Problem
- Linear programming formulation
- Monge vs. Kantorovich perspective
- The Earth Mover's Distance (EMD)
- Computational aspects: simplex, Hungarian algorithm

### Module 2: Theory & Metrics
- The Wasserstein distance $W_p$
- Metric properties and triangle inequality
- Closed-form solutions for Gaussians (Bures formula)
- Connections to $c$-transforms and duality

### Module 3: Duality & Optimality Conditions
- Kantorovich dual problem
- Kantorovich potentials $(\varphi, \psi)$
- $c$-cyclical monotonicity
- Characterizing optimal transport plans

### Module 4: Entropy-Regularized OT (Sinkhorn)
- Motivation: computational speed vs. regularization bias
- Naive Sinkhorn algorithm (convergence and numerical issues)
- Stabilized log-domain Sinkhorn
- Applications to machine learning

### Module 5: Monge Problem & Brenier Maps
- The monge map (optimal transport law)
- Existence, uniqueness, and regularity
- Brenier's theorem
- Computational approximation

### Module 6: Applications
- **Generative Modeling**: Wasserstein Autoencoders (WGAN)
- **Domain Adaptation**: Aligning distributions for transfer learning
- **Barycenter Problems**: Computing Fréchet means of distributions
- **Shape Matching**: Comparing objects in metric space

### Module 7: Advanced Topics
- Unbalanced optimal transport
- Sliced and partial OT
- Multi-marginal problems
- Scalability and approximations

### Module 8: Connections & Perspectives
- OT in machine learning (summary)
- Gradient flows and PDEs
- Open problems and research directions

## 📝 Exercises

The `optimal-transport-exercises.ipynb` notebook contains **12 structured exercises**:

| # | Title | Module | Concepts |
|---|-------|--------|----------|
| 1 | Discrete Kantorovich by hand | 1.3 | LP formulation, vectorization, linprog |
| 2 | 1D comonotone coupling | 1 | Sorting shortcut, W₂ in 1D |
| 3 | Gaussian W₂ & Bures formula | 2 | Matrix square roots, empirical bias |
| 4 | Geometry: W₁ vs. KL | 0 | Divergence comparison, gradient flows |
| 5 | Triangle inequality (empirical) | 2.1 | Metric properties, triangle inequality slack |
| 6 | Kantorovich potentials | 3.1, 3.3 | Dual LP, complementary slackness |
| 7 | Sinkhorn from scratch | 4.2 | Algorithm implementation, convergence |
| 8 | Stabilized log-domain Sinkhorn | 4.2 | Numerical stability, regularization path |
| 9 | Monge maps & Brenier theorem | 5 | Gradient computation, Monge pushforward |
| 10 | Classification with OT | 6 | Domain adaptation, supervised OT |
| 11 | Wasserstein Barycenter | 6 | Multi-marginal OT, iterative algorithms |
| 12 | WGAN on toy data | 6 | Generative modeling, neural networks |

### How to Use the Exercises

- **Work in order**: Exercises build on each other
- **Code scaffolds** are provided; fill in the `# YOUR CODE HERE` sections
- **Reference solutions** exist but writing your own is the learning goal
- **Estimated time**: 10–15 hours total, depending on depth
- **If stuck for >30 minutes**: move on and return later

## 🛠️ Setup & Installation

### Requirements

- **Python 3.8+**
- **Core libraries**:
  ```bash
  pip install numpy scipy matplotlib scikit-learn
  ```
- **Optimal Transport library** (essential):
  ```bash
  pip install POT  # Python Optimal Transport
  ```
- **Optional** (for advanced exercises):
  ```bash
  pip install torch torchvision  # For WGAN exercise
  ```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/MohamedBaha/OptimalTransport_CrashCourse.git
cd OptimalTransport_CrashCourse

# Install dependencies
pip install -r requirements.txt  # (if available)

# Launch Jupyter
jupyter notebook

# Start with the main course
open optimal-transport-course.ipynb

# Work through exercises
open optimal-transport-exercises.ipynb
```

## 📐 Mathematical Notation

Used consistently throughout the course:

| Symbol | Meaning |
|--------|---------|
| $\mu, \nu$ | Probability measures on $\mathbb{R}^d$ |
| $a \in \Delta^{n-1}$, $b \in \Delta^{m-1}$ | Discrete probability weights |
| $C \in \mathbb{R}_{+}^{n \times m}$ | Cost matrix |
| $P$ | Transport plan (coupling matrix) |
| $W_p(\mu, \nu)$ | Wasserstein-$p$ distance with cost $c(x,y) = \|x - y\|^p$ |
| $(\varphi, \psi)$ | Kantorovich dual potentials |
| $\varepsilon$ | Entropic regularization parameter |

## 🔗 Key Resources

### Textbooks & Surveys
- Peyré, Cuturi (2019): *Computational Optimal Transport* – comprehensive treatment
- Santambrogio (2015): *Optimal Transport for Applied Mathematicians*
- Villani (2008): *Optimal Transport: Old and New* – mathematical foundations

### Software
- [POT (Python Optimal Transport)](https://pythonot.github.io/) – main library used here
- [Gromov-Wasserstein library](https://github.com/mionakowski/gromov_wasserstein)
- JAX/Diffrax implementations for GPU acceleration

### Recent Papers
- Wasserstein Autoencoders (Tolstikhin et al., 2017)
- Sliced Wasserstein Distances (Bonneel et al., 2015)
- Conditional Mean Embeddings via OT (Courty et al.)

## 💡 Key Takeaways

After completing this course, you will:

1. **Understand OT fundamentals**: Monge problem, Kantorovich formulation, duality
2. **Implement algorithms**: Exact (EMD) and approximate (Sinkhorn) solvers
3. **Apply OT in practice**: Domain adaptation, barycenter computation, shape matching
4. **Recognize failure modes**: Numerical instability, high-dimensional scaling, regularization trade-offs
5. **Connect to applications**: GANs, metric learning, transfer learning

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional exercises or solutions
- GPU-accelerated implementations
- Visualization improvements
- Corrections or clarifications
- New application examples

Please open an issue or pull request to contribute.

## 📄 License

This repository is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Mohamed Baha**  
GitHub: [@MohamedBaha](https://github.com/MohamedBaha)

## 🙏 Acknowledgments

- Gabriel Peyré & Marco Cuturi for foundational texts on computational OT
- The [POT library](https://pythonot.github.io/) developers
- Feedback from students and collaborators

---

**Last Updated**: May 2026  
**Course Status**: Active development – exercises and materials are continuously improved

