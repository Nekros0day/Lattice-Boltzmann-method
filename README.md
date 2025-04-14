# Phase Field LBM 2D Simulation

This repository provides a Python implementation of a phase-field simulation in two dimensions. The simulation couples diffusion (modeled by the Lattice Boltzmann Method) with nonlinear reaction kinetics (including several versions of the Allen–Cahn reaction) to study interface dynamics in reaction-diffusion systems.

## Overview

The code implements:
- **LBM Diffusion:** Utilizing the D2Q9 lattice model, the simulation solves a diffusion problem based on a relaxation scheme where the LBM distribution functions are updated and streamed on a 2D grid.
- **Reaction Kinetics:** The simulation supports multiple reaction terms:
  - *Standard Allen–Cahn*: Uses \((\phi^3 - \phi)/\epsilon^2\).
  - *Mass-Conserving Allen–Cahn*: Adjusts the reaction term by subtracting its spatial mean to conserve mass.
- **Initial Conditions:** A default smooth oval initial condition is provided, with the option to load a user-supplied condition from a file.

The simulation parameters such as grid size, number of time steps, and reaction type are all configurable, making it easy to adapt for different research or educational needs.
