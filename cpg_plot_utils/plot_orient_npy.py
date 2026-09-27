import numpy as np
import matplotlib.pyplot as plt

orient_vmc = np.load("raw_data_plot/orientation_vmc.npy")  # shape (3, N)
orient_novmc = np.load("raw_data_plot/orientation_novmc.npy")

# Time vector (adjust dt if needed)
dt = 0.001
t = np.arange(orient_vmc.shape[1]) * dt

# Global style (match trajectory plot)
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
})

fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
fig.suptitle("Body Orientation", fontsize=14)

labels = ["Roll", "Pitch"]
vmc_color = "#1f77b4"      # muted blue
novmc_color = "#d62728"    # muted red

for i, ax in enumerate(axs):
    ax.plot(t, orient_vmc[i, :], color=vmc_color, linewidth=2, label="VMC")
    ax.plot(t, orient_novmc[i, :], color=novmc_color, linestyle="--", linewidth=2, label="No VMC")
    ax.set_ylabel(f"{labels[i]} [rad]")
    
    # Subtle grid
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    
    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axs[-1].set_xlabel("Time [s]")
axs[-1].set_xlim(t[0], t[-1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
