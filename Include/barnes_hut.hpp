#pragma once

#include "layout_engine.hpp"
#include "quadtree.hpp"

#include <glm/geometric.hpp>
#include <span>
#include <algorithm>
#include <limits>

// ============================================================
//  BarnesHutRepulsion  –  O(|V| log |V|) repulsive strategy
// ============================================================

/**
 * Implements IRepulsiveStrategy using the Barnes-Hut multipole
 * acceptance criterion.
 *
 * Algorithm per step():
 *   1. Compute a tight BoundingBox around all current node positions.
 *   2. Build a QuadTree by inserting every node — O(|V| log |V|).
 *   3. For each node v, traverse the tree:
 *        - If cell is a leaf containing only v itself  → skip (self-force).
 *        - If s/d < θ (Barnes-Hut criterion)          → treat cell as
 *          a single super-node at its center of mass.
 *        - Otherwise                                   → recurse into children.
 *      Each accepted interaction computes:
 *        F_r = (k² / d²) * δ,    δ = v.pos - cell.CoM
 *      This is the same formula as brute-force; the approximation lies in
 *      replacing many individual nodes with one aggregate node.
 *
 * Complexity:
 *   Tree construction : O(|V| log |V|)
 *   Force computation : O(|V| log |V|)   expected for θ ∈ (0,1)
 *   Total             : O(|V| log |V|)   vs O(|V|²) for BruteForce
 *
 * Parameter θ (theta):
 *   Controls accuracy vs speed trade-off.
 *   θ = 0.0  →  exact (degenerates to O(|V|²))
 *   θ = 0.5  →  standard choice (good balance for layout)
 *   θ = 1.0  →  very aggressive approximation (faster, less accurate)
 */
class BarnesHutRepulsion final : public IRepulsiveStrategy {
public:
    explicit BarnesHutRepulsion(float theta = 0.5f) noexcept
        : theta_(theta) {}

    void setTheta(float theta) noexcept { theta_ = theta; }
    [[nodiscard]] float theta() const noexcept { return theta_; }

    // ── IRepulsiveStrategy interface ──────────────────────────

    void computeRepulsive(std::span<Node> nodes, float k) override {
        if (nodes.empty()) return;

        // ── 1. Build tight bounding box ───────────────────────
        BoundingBox bounds = computeBounds(nodes);

        // ── 2. Build QuadTree ─────────────────────────────────
        QuadTree tree{ bounds };
        for (const Node& v : nodes)
            tree.insert(v.position, v.id);

        // ── 3. Compute repulsive force for each node ──────────
        const float k2 = k * k;
        for (Node& v : nodes)
            v.displacement += forceFromTree(tree, v.position, v.id, k2);
    }

private:
    float theta_;

    // ── Bounds helper ─────────────────────────────────────────

    static BoundingBox computeBounds(std::span<const Node> nodes) noexcept {
        float minX =  std::numeric_limits<float>::max();
        float minY =  std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();

        for (const Node& v : nodes) {
            minX = std::min(minX, v.position.x);
            minY = std::min(minY, v.position.y);
            maxX = std::max(maxX, v.position.x);
            maxY = std::max(maxY, v.position.y);
        }

        // Add a small margin to avoid boundary precision issues
        const float margin = 1.0f;
        const glm::vec2 center{ (minX + maxX) * 0.5f,
                                (minY + maxY) * 0.5f };
        const float halfW = (maxX - minX) * 0.5f + margin;
        const float halfH = (maxY - minY) * 0.5f + margin;

        return BoundingBox{ center, halfW, halfH };
    }

    // ── Recursive tree traversal ──────────────────────────────

    /**
     * Computes the net repulsive force on a node at position `pos`
     * (with id `selfId`) by walking the QuadTree.
     *
     * Barnes-Hut criterion:  s / d < θ
     *   s = cell size  (longest side of the bounding box)
     *   d = distance from pos to cell center of mass
     *
     * When the criterion is met the entire subtree contributes a
     * single force proportional to its total mass (number of nodes):
     *
     *   F_r = totalMass * k² / d²  * δ_unit
     *
     * The factor `totalMass` arises because each node contributes
     * an independent repulsive force of magnitude k²/d², and within
     * the accepted cell all nodes are approximated as co-located at
     * the center of mass.
     */
    [[nodiscard]] glm::vec2 forceFromTree(const QuadTree& node,
                                          glm::vec2        pos,
                                          std::uint32_t    selfId,
                                          float            k2) const
    {
        if (node.totalMass() == 0) return { 0.0f, 0.0f };

        glm::vec2 delta = pos - node.centerOfMass();
        float dist      = glm::length(delta);

        // ── Self-exclusion for exact leaf ─────────────────────
        if (node.isLeaf()) {
            if (node.totalMass() == 1 && node.leafId() == selfId)
                return { 0.0f, 0.0f };                // skip self
        }

        // ── Guard against coincident positions ────────────────
        if (dist < 1e-4f) {
            dist  = 1e-4f;
            delta = glm::vec2{ 1e-4f, 0.0f };
        }

        // ── Barnes-Hut acceptance criterion: s/d < θ ─────────
        // s = node.bounds().size()  (longest side of cell)
        // d = dist                  (distance to center of mass)
        //
        // When accepted, the force is scaled by totalMass because
        // each constituent graph node contributes k²/d² independently.
        const float s = node.bounds().size();
        if (node.isLeaf() || (s / dist) < theta_) {
            // Treat entire subtree as one super-node at CoM
            const float mass    = static_cast<float>(node.totalMass());
            const float forceMag = mass * k2 / (dist * dist);
            return (delta / dist) * forceMag;
        }

        // ── Recurse into children ─────────────────────────────
        glm::vec2 total{ 0.0f, 0.0f };
        for (const auto& child : node.children())
            if (child) total += forceFromTree(*child, pos, selfId, k2);
        return total;
    }
};
