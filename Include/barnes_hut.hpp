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
 * Barnes-Hut multipole approximation for repulsive forces.
 *
 * Per call to computeRepulsive():
 *   1. Compute tight BoundingBox around all node positions.
 *   2. Build a pool-based QuadTree — O(|V| log |V|), zero heap allocs
 *      after the first call (pool memory is reused via reset()).
 *   3. For each node v, walk the tree:
 *        - Leaf containing only v itself  → skip (self-force).
 *        - s / d < θ                      → accept: treat subtree as
 *          a single super-node at its centre of mass.
 *        - Otherwise                      → recurse into children.
 *      Accepted force:
 *        F_r = totalMass * k² / d²  * (δ / |δ|)
 *      where δ = v.pos − cell.CoM.
 *      The totalMass factor accounts for each constituent node
 *      contributing an independent repulsion of magnitude k²/d².
 *
 * Complexity:
 *   Tree build  : O(|V| log |V|)
 *   Force query : O(|V| log |V|)  expected for θ ∈ (0,1)
 *
 * θ trade-off:
 *   θ = 0.0 → exact O(|V|²)   θ = 0.5 → standard   θ = 1.0 → aggressive
 */
class BarnesHutRepulsion final : public IRepulsiveStrategy {
public:
    explicit BarnesHutRepulsion(float theta = 0.5f) noexcept
        : theta_(theta),
          tree_(BoundingBox{{ 0,0 }, 1, 1 }, 512)  // placeholder; reset each call
    {}

    void setTheta(float theta) noexcept { theta_ = theta; }
    [[nodiscard]] float theta() const noexcept { return theta_; }

    // ── IRepulsiveStrategy ────────────────────────────────────

    void computeRepulsive(std::span<Node> nodes, float k) override {
        if (nodes.empty()) return;

        // ── 1. Tight bounding box ─────────────────────────────
        BoundingBox bounds = computeBounds(nodes);

        // ── 2. Build QuadTree (reuses pool memory) ────────────
        tree_.reset(bounds);
        for (const Node& v : nodes)
            tree_.insert(v.position, v.id);

        // ── 3. Repulsive force per node ───────────────────────
        const float k2 = k * k;
        for (Node& v : nodes)
            v.displacement += queryNode(0, v.position, v.id, k2);
    }

private:
    float     theta_;
    QuadTree  tree_;   // persists across calls — pool reused each iteration

    // ── Bounds ────────────────────────────────────────────────

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

        const float margin = 1.0f;
        return BoundingBox{
            { (minX + maxX) * 0.5f, (minY + maxY) * 0.5f },
            (maxX - minX) * 0.5f + margin,
            (maxY - minY) * 0.5f + margin
        };
    }

    // ── Recursive tree walk ───────────────────────────────────

    [[nodiscard]] glm::vec2 queryNode(int          nodeIdx,
                                      glm::vec2    pos,
                                      std::uint32_t selfId,
                                      float         k2) const
    {
        const QuadTree::Node& n = tree_.at(nodeIdx);
        if (n.totalMass < 0.5f) return { 0.0f, 0.0f };

        glm::vec2 delta = pos - n.centerOfMass;
        float dist      = glm::length(delta);

        // Self-exclusion at exact leaf
        if (n.isLeaf()) {
            if (n.hasPoint && n.pointId == selfId)
                return { 0.0f, 0.0f };
        }

        if (dist < 1e-4f) {
            dist  = 1e-4f;
            delta = glm::vec2{ 1e-4f, 0.0f };
        }

        // Barnes-Hut criterion: s / d < θ
        const float s = n.bounds.size();
        if (n.isLeaf() || (s / dist) < theta_) {
            const float forceMag = n.totalMass * k2 / (dist * dist);
            return (delta / dist) * forceMag;
        }

        // Recurse into children
        glm::vec2 total{ 0.0f, 0.0f };
        for (int q = 0; q < 4; ++q) {
            int ci = n.children[q];
            if (ci != QuadTree::NULL_NODE)
                total += queryNode(ci, pos, selfId, k2);
        }
        return total;
    }
};
