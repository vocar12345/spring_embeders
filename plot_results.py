import pandas as pd
import matplotlib.pyplot as plt

# Convergence curve
metrics = pd.read_csv("output/metrics.csv")
plt.figure(figsize=(8, 4))
plt.semilogy(metrics["iteration"], metrics["kinetic_energy"])
plt.xlabel("Iteration")
plt.ylabel("Kinetic Energy (log scale)")
plt.title("Fruchterman-Reingold Convergence")
plt.tight_layout()
plt.savefig("convergence.pdf")

# Graph layout
nodes = pd.read_csv("output/nodes.csv")
edges = pd.read_csv("output/edges.csv")

fig, ax = plt.subplots(figsize=(10, 6))
for _, e in edges.iterrows():
    u = nodes[nodes.node_id == e.source].iloc[0]
    v = nodes[nodes.node_id == e.target].iloc[0]
    ax.plot([u.x, v.x], [u.y, v.y], "b-", lw=0.6, alpha=0.5)
ax.scatter(nodes.x, nodes.y, s=40, zorder=3)
plt.savefig("layout.pdf")