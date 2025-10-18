# 🔥 Heat Distribution Simulation in a Square Rod

This project presents a **numerical simulation** that models the steady-state heat distribution in an infinitely long rod with a square cross-section.  
It demonstrates the application of **finite difference methods** and **Neumann boundary conditions** to approximate temperature fields and heat flux, verified with **Gauss’ theorem**.

---

## 🧠 Project Overview
The goal of this project was to simulate how heat distributes within a solid rod over time, assuming constant boundary temperatures.  
The numerical approach is based on:
- Implementation of **Neumann (second-kind) boundary conditions**,
- An **iterative Jacobi method** for convergence,
- Visualization of the **temperature field** and **heat flux vectors**,
- Validation of results using **Gauss' theorem**.

---

## ⚙️ Technologies
- **Python 3.x**
- **NumPy** – numerical operations and grid computation  
- **SciPy** – linear algebra and gradient calculation  
- **Matplotlib** – 2D/3D visualization and contour plots  

---

## 📋 Model Description

### Constants and Initial Conditions
The rectangular simulation domain (`8 m × 6 m`) is discretized into a grid:

```python
Nx = 100  # number of points along x-axis
Ny = 90   # number of points along y-axis
