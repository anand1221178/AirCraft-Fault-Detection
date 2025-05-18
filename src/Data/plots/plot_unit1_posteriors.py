import pandas as pd
import matplotlib.pyplot as plt

# Load posterior CSV 
df = pd.read_csv("unit1_posteriors.csv")

# Plot settings
plt.figure(figsize=(10, 5))
plt.plot(df["cycle"], df["P(Engine_Core_Health=Healthy)"], label="Healthy", color="green")
plt.plot(df["cycle"], df["P(Engine_Core_Health=Degrading)"], label="Degrading", color="orange")
plt.plot(df["cycle"], df["P(Engine_Core_Health=Critical)"], label="Critical", color="red")

# Ground truth background coloring 
if "true" in df.columns:
    for state in df["true"].unique():
        mask = df["true"] == state
        plt.fill_between(df["cycle"], 0, 1, where=mask, alpha=0.05, transform=plt.gca().get_xaxis_transform(),
                         label=f"GT: {state}")

plt.title("Posterior Marginals for Unit 1")
plt.xlabel("Cycle")
plt.ylabel("P(Health State)")
plt.legend()
plt.tight_layout()
plt.savefig("unit1_posteriors.png", dpi=300)
plt.show()
