import numpy as np
import matplotlib.pyplot as plt

# === Default Reaction Functions ===
def reaction_allen_cahn(phi, epsilon):
    """Standard Allen–Cahn reaction term: (phi^3 - phi)/epsilon^2."""
    return (phi**3 - phi) / (epsilon**2)

def reaction_mass_conserving(phi, epsilon):
    """
    Mass-conserving Allen–Cahn reaction term:
    (phi^3 - phi) with its spatial mean subtracted, divided by epsilon^2.
    """
    term = phi**3 - phi
    term -= np.mean(term)
    return term / (epsilon**2)

def reaction_logistic(phi, epsilon):
    """
    Sample logistic reaction term.
    (Note: this reaction term does not use epsilon.)
    """
    return phi * (1 - phi)

# === Main Solver Function ===
def solve_phase_field_lbm_2d(
    nx=100,
    ny=100,
    max_steps=3000,
    tau=1.0,
    epsilon=1.0,
    a=20,
    b=10,            # Default: ellipse semi-axes for the initial condition.
    num_frames=8,
    smoothing_delta=0.1,
    initial_condition_file=None,
    reaction_type="allen_cahn",
    reaction_fn=None
):
    """
    Solve a phase-field reaction-diffusion problem in 2D using LBM for diffusion 
    plus an explicit reaction step.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    max_steps : int
        Total LBM timesteps.
    tau : float
        LBM relaxation time (sets diffusion: D = (tau - 0.5)/3).
    epsilon : float
        Parameter affecting interface thickness in the reaction term.
    a, b : float
        Semi-axes for the default (oval) initial condition.
    num_frames : int
        Number of snapshots to record.
    smoothing_delta : float
        Controls the smoothness of the interface in the default initial condition.
    initial_condition_file : str or None
        Path to a file (e.g., a NumPy .npy file) containing a 2D initial condition.
        If None, a default smooth oval is used.
    reaction_type : str
        One of "allen_cahn" or "mass_conserving". Used if reaction_fn is None.
    reaction_fn : function or None
        User-supplied function f(phi, epsilon) returning the reaction term.
        If None, one is chosen based on reaction_type.
        
    Returns
    -------
    frames : list of 2D numpy.ndarray
        Saved snapshots of the phase field.
    times : list of int
        Time steps corresponding to the snapshots.
    """
    # === Discrete Velocities and Weights for D2Q9 ===
    c = np.array([
        [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, 1], [-1, -1], [1, -1]
    ], dtype=np.int32)
    w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    
    # === Initialize the Phase Field (phi) ===
    if initial_condition_file is not None:
        # Load initial condition from file; expect a 2D array with shape (ny, nx)
        phi = np.load(initial_condition_file)
        if phi.shape != (ny, nx):
            raise ValueError(f"Loaded initial condition has shape {phi.shape},"
                             f" but expected ({ny}, {nx}).")
    else:
        # Default: Create a smooth oval using a tanh transition.
        x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny))
        cx0, cy0 = nx / 2, ny / 2
        ellipse_region = ((x_grid - cx0)**2 / a**2 + (y_grid - cy0)**2 / b**2)
        phi = np.tanh((1 - ellipse_region) / smoothing_delta)
    
    # === Initialize the LBM Distribution Function ===
    f = np.zeros((ny, nx, 9))
    for i in range(9):
        f[:, :, i] = w[i] * phi

    # The effective diffusion coefficient (for reference) is:
    # D = (tau - 0.5)/3, which is implicitly used via tau.
    
    # === Setup for Saving Snapshots ===
    save_interval = max_steps // (num_frames - 1) if num_frames > 1 else max_steps
    frames = []
    times = []
    def save_frame(step):
        frames.append(phi.copy())
        times.append(step)
    save_frame(0)
    
    # === Select Reaction Function ===
    if reaction_fn is None:
        if reaction_type == "mass_conserving":
            reaction_fn = reaction_mass_conserving
        else:
            reaction_fn = reaction_allen_cahn  # default
    
    # === Main Time-Stepping Loop (LBM Diffusion + Reaction) ===
    for step in range(1, max_steps + 1):
        # Collision: relax toward equilibrium f_eq = w * phi
        f_eq = np.zeros_like(f)
        for i in range(9):
            f_eq[:, :, i] = w[i] * phi
        f_coll = f - (1.0 / tau) * (f - f_eq)
        
        # Streaming: shift the collided distribution along lattice directions
        f_new = np.zeros_like(f)
        for i in range(9):
            cx, cy = c[i]
            f_shifted = np.roll(f_coll[:, :, i], shift=cy, axis=0)
            f_shifted = np.roll(f_shifted, shift=cx, axis=1)
            f_new[:, :, i] = f_shifted
        f = f_new
        
        # Recompute phi from the distribution functions
        phi = np.sum(f, axis=2)
        
        # Apply the reaction term (explicit Euler update)
        reaction_term = reaction_fn(phi, epsilon)
        phi = phi - reaction_term
        
        # Redistribute the updated phi into f
        for i in range(9):
            f[:, :, i] = w[i] * phi
        
        # Save snapshots at specified intervals
        if step % save_interval == 0 and len(frames) < num_frames:
            save_frame(step)
    
    return frames, times

# === Demo Function ===
def demo_phase_field_lbm_2d():
    """
    Run the LBM-based phase field simulation and produce a figure with subplots
    showing the evolution of phi. All subplots share a consistent color scale and
    a common colorbar is placed on the far right.
    """
    # Simulation parameters (user adjustable)
    nx, ny = 120, 120
    max_steps = 3000
    tau = 1.0
    epsilon = 1.0
    a, b = 30, 15
    num_frames = 8
    smoothing_delta = 0.1

    # Choose a reaction type: options "allen_cahn", "mass_conserving"
    reaction_type = "allen_cahn"   # Change as desired.
    
    # Optionally, specify an initial condition file (set to None to use default oval)
    initial_condition_file = None

    # Run the simulation
    frames, times = solve_phase_field_lbm_2d(
        nx=nx,
        ny=ny,
        max_steps=max_steps,
        tau=tau,
        epsilon=epsilon,
        a=a,
        b=b,
        num_frames=num_frames,
        smoothing_delta=smoothing_delta,
        initial_condition_file=initial_condition_file,
        reaction_type=reaction_type
    )
    
    # Plot settings (consistent z-scale)
    vmin, vmax = -1, 1
    cmap = "viridis"
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        if i < len(frames):
            im = ax.imshow(frames[i], origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Timestep = {times[i]}")
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$\phi$ (Order Parameter)")
    
    plt.savefig("phase_field.png")
    plt.show()

if __name__ == "__main__":
    demo_phase_field_lbm_2d()
