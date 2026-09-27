import numpy as np
import matplotlib.pyplot as plt

traj_vmc = np.load("cpg_plot_utils/traj_vmc.npy")
traj_novmc = np.load("cpg_plot_utils/traj_novmc.npy")

# Use a clean, professional style
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
})

fig, ax = plt.subplots(figsize=(6.5, 5))

# Professional color palette
vmc_color = "#1f77b4"      # muted blue
novmc_color = "#d62728"    # muted red

# Trajectories
ax.plot(
    traj_vmc[0, :], traj_vmc[1, :],
    color=vmc_color, linewidth=2.5, label="VMC"
)
ax.plot(
    traj_novmc[0, :], traj_novmc[1, :],
    color=novmc_color, linewidth=2.5, linestyle="--", label="No VMC"
)

# End points
ax.scatter(
    traj_vmc[0, -1], traj_vmc[1, -1],
    marker="x", s=80, linewidths=2, color=vmc_color, zorder=5
)
ax.scatter(
    traj_novmc[0, -1], traj_novmc[1, -1],
    marker="x", s=80, linewidths=2, color=novmc_color, zorder=5
)

# Labels and formatting
ax.set_title("Body XY Trajectory")
ax.set_xlabel("X Position [m]")
ax.set_ylabel("Y Position [m]")
ax.set_aspect("equal", adjustable="box")

# Subtle grid
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

# Remove top/right spines for a cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
ax.legend(frameon=False, loc="best")

plt.tight_layout()
plt.show()

