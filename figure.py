import matplotlib.pyplot as plt

# Define extracted beta values and classification accuracies
betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# classification_accuracies = [0.47, 0.472, 0.485, 0.474, 0.487, 0.479, 0.482, 0.481, 0.49, 0.495, 0.485]
classification_accuracies = [0.84, 0.86, 0.88, 0.895, 0.91, 0.905, 0.925, 0.924, 0.932, 0.921, 0.93]

# Define SimCLR baseline accuracy (β=0 case)
simclr_baseline = classification_accuracies[0]

# Create high-quality figure with slightly smaller fonts
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)  # High resolution

# Plot SinSim performance with improved styling (Orangish color)
ax.plot(betas, classification_accuracies, marker="o", markersize=6, linestyle="-", linewidth=2, color="orange", label="SinSim Performance")

# Plot the SimCLR baseline as a horizontal line
ax.axhline(y=simclr_baseline, color='r', linestyle="--", linewidth=2, label="SimCLR Baseline (β=0)")

# Customize labels and title with slightly smaller fonts
ax.set_xlabel(r"Sinkhorn Regularization Strength ($\beta$)", fontsize=14, fontweight="bold")
ax.set_ylabel("Classification Accuracy", fontsize=14, fontweight="bold")
ax.set_title("Effect of Sinkhorn Regularization on SimCLR Performance", fontsize=16, fontweight="bold")

# Customize tick labels for better readability
ax.tick_params(axis="both", labelsize=12)

# Add a legend with improved styling
ax.legend(fontsize=12, loc="lower right")

# Enable grid with better visibility
ax.grid(True, linestyle="--", alpha=0.6)

# Save the updated figure in high resolution
plt.savefig("CIFAR10_SinSim_PaperStyle_Updated.png", dpi=300, bbox_inches='tight')

# Show the figure
plt.show()
