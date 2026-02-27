#pragma once

#include <glm/glm.hpp>
#include <array>
#include <memory>
#include <vector>
#include <cassert>

// ============================================================
//  BoundingBox  –  axis-aligned 2D rectangle
// ============================================================

struct BoundingBox {
    glm::vec2 center;   // geometric centre of this cell
    float     halfW;    // half-width  (x extent)
    float     halfH;    // half-height (y extent)

    [[nodiscard]] bool contains(glm::vec2 p) const noexcept {
        return p.x >= center.x - halfW && p.x <= center.x + halfW
            && p.y >= center.y - halfH && p.y <= center.y + halfH;
    }

    /// Returns one of four sub-quadrants (0=NE,1=NW,2=SW,3=SE).
    [[nodiscard]] std::array<BoundingBox, 4> subdivide() const noexcept {
        const float qW = halfW * 0.5f;
        const float qH = halfH * 0.5f;
        return {{
            { { center.x + qW, center.y + qH }, qW, qH },  // NE
            { { center.x - qW, center.y + qH }, qW, qH },  // NW
            { { center.x - qW, center.y - qH }, qW, qH },  // SW
            { { center.x + qW, center.y - qH }, qW, qH },  // SE
        }};
    }

    /// Longest side length — used for the Barnes-Hut s/d criterion.
    [[nodiscard]] float size() const noexcept {
        return 2.0f * std::max(halfW, halfH);
    }
};

// ============================================================
//  QuadTree
// ============================================================

/**
 * A point-region QuadTree storing one graph node per leaf.
 *
 * Aggregated quantities per internal node:
 *   centerOfMass  – mass-weighted average position of all contained nodes
 *   totalMass     – number of graph nodes in this subtree (all nodes have
 *                   equal "mass" = 1 in the layout context)
 *
 * Relation to optimal distance k:
 *   The Barnes-Hut criterion compares s/d against θ, where:
 *     s = cell.size()   (spatial width of this QuadTree cell)
 *     d = ||v - centerOfMass||  (distance from query node to cell CoM)
 *   When s/d < θ, the entire subtree is treated as a single super-node
 *   located at centerOfMass.  The repulsive force is then:
 *     F_r = k² / d²  * δ   (same formula as brute-force, but against CoM)
 *   This approximation is valid because nodes far from the cell contribute
 *   nearly-identical force directions, and k sets the length scale at which
 *   repulsion becomes negligible — so cells where s << d can be collapsed
 *   without introducing significant error into the layout.
 */
class QuadTree {
public:
    // Maximum graph nodes in a leaf before it splits.
    static constexpr int CAPACITY = 1;

    // ── Construction ─────────────────────────────────────────
    explicit QuadTree(BoundingBox bounds) noexcept
        : bounds_(bounds) {}

    // Non-copyable; move is fine.
    QuadTree(const QuadTree&)            = delete;
    QuadTree& operator=(const QuadTree&) = delete;
    QuadTree(QuadTree&&)                 = default;
    QuadTree& operator=(QuadTree&&)      = default;

    // ── Insertion ────────────────────────────────────────────

    /**
     * Inserts a point (graph node position) with given index.
     * Updates centerOfMass and totalMass on the path to the leaf.
     *
     * @param pos    Position of the graph node.
     * @param nodeId Identifier stored at the leaf (for self-exclusion).
     * @return       false if pos lies outside this cell's bounds.
     */
    bool insert(glm::vec2 pos, std::uint32_t nodeId) {
        if (!bounds_.contains(pos)) return false;

        // ── Update aggregate (online mean formula) ────────────
        centerOfMass_ = (centerOfMass_ * static_cast<float>(totalMass_) + pos)
                        / static_cast<float>(totalMass_ + 1);
        ++totalMass_;

        // ── Leaf with room ────────────────────────────────────
        if (!isSubdivided_ && totalMass_ <= CAPACITY) {
            point_   = pos;
            pointId_ = nodeId;
            return true;
        }

        // ── First overflow: subdivide and re-insert old point ─
        if (!isSubdivided_) {
            subdivide();
            // Re-insert the point that was already here
            insertIntoChildren(point_, pointId_);
        }

        insertIntoChildren(pos, nodeId);
        return true;
    }

    // ── Accessors ────────────────────────────────────────────
    [[nodiscard]] glm::vec2    centerOfMass() const noexcept { return centerOfMass_; }
    [[nodiscard]] int          totalMass()    const noexcept { return totalMass_;    }
    [[nodiscard]] bool         isLeaf()       const noexcept { return !isSubdivided_; }
    [[nodiscard]] const BoundingBox& bounds() const noexcept { return bounds_;       }

    [[nodiscard]] const std::array<std::unique_ptr<QuadTree>, 4>&
    children() const noexcept { return children_; }

    /// The single point stored in a leaf node.
    [[nodiscard]] glm::vec2    leafPoint()  const noexcept { return point_;   }
    [[nodiscard]] std::uint32_t leafId()    const noexcept { return pointId_; }

private:
    // ── Helpers ───────────────────────────────────────────────

    void subdivide() {
        auto quads = bounds_.subdivide();
        for (int i = 0; i < 4; ++i)
            children_[i] = std::make_unique<QuadTree>(quads[i]);
        isSubdivided_ = true;
    }

    void insertIntoChildren(glm::vec2 pos, std::uint32_t id) {
        for (auto& child : children_)
            if (child && child->insert(pos, id)) return;
    }

    // ── Data ──────────────────────────────────────────────────
    BoundingBox bounds_;

    // Aggregated quantities (valid for both internal and leaf nodes)
    glm::vec2    centerOfMass_{ 0.0f, 0.0f };
    int          totalMass_   { 0           };

    // Leaf storage (only meaningful when isLeaf() == true)
    glm::vec2    point_   { 0.0f, 0.0f };
    std::uint32_t pointId_{ 0           };

    bool isSubdivided_{ false };
    std::array<std::unique_ptr<QuadTree>, 4> children_;
};
