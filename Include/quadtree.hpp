#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

// ============================================================
//  BoundingBox
// ============================================================

struct BoundingBox {
    glm::vec2 center;
    float     halfW;
    float     halfH;

    [[nodiscard]] bool contains(glm::vec2 p) const noexcept {
        return p.x >= center.x - halfW && p.x <= center.x + halfW
            && p.y >= center.y - halfH && p.y <= center.y + halfH;
    }

    [[nodiscard]] float size() const noexcept {
        return 2.0f * std::max(halfW, halfH);
    }

    [[nodiscard]] BoundingBox child(int q) const noexcept {
        const float qW = halfW * 0.5f;
        const float qH = halfH * 0.5f;
        switch (q) {
            case 0: return {{ center.x + qW, center.y + qH }, qW, qH }; // NE
            case 1: return {{ center.x - qW, center.y + qH }, qW, qH }; // NW
            case 2: return {{ center.x - qW, center.y - qH }, qW, qH }; // SW
            default:return {{ center.x + qW, center.y - qH }, qW, qH }; // SE
        }
    }

    [[nodiscard]] int quadrant(glm::vec2 p) const noexcept {
        const bool right = p.x >= center.x;
        const bool up    = p.y >= center.y;
        if  (right  &&  up) return 0;
        if  (!right &&  up) return 1;
        if  (!right && !up) return 2;
        return 3;
    }
};

// ============================================================
//  QuadTree  –  flat pool allocator, zero per-iteration heap
// ============================================================

/**
 * Pool-based QuadTree with bounding box export support.
 *
 * The collectBoxes() method recursively walks the tree and collects
 * bounding boxes of all internal cells — used to visualise the spatial
 * subdivision structure overlaid on the final graph layout.
 *
 * Relation to optimal distance k:
 *   The acceptance criterion s/d < θ compares cell size s to the
 *   distance d from query node to cell CoM. Both are in the same
 *   spatial units as k, so the criterion is scale-invariant.
 */
class QuadTree {
public:
    static constexpr int NULL_NODE = -1;

    struct Node {
        BoundingBox   bounds;
        glm::vec2     centerOfMass{ 0.0f, 0.0f };
        float         totalMass   { 0.0f };
        glm::vec2     point       { 0.0f, 0.0f };
        std::uint32_t pointId     { 0           };
        bool          hasPoint    { false        };
        int           children[4] { NULL_NODE, NULL_NODE, NULL_NODE, NULL_NODE };

        [[nodiscard]] bool isLeaf() const noexcept {
            return children[0] == NULL_NODE;
        }
    };

    // ── Construction ─────────────────────────────────────────

    explicit QuadTree(BoundingBox bounds, std::size_t expectedNodes = 512) {
        pool_.reserve(expectedNodes * 4);
        pool_.push_back(Node{ bounds });
    }

    void reset(BoundingBox bounds) {
        pool_.clear();
        pool_.push_back(Node{ bounds });
    }

    // ── Insertion ─────────────────────────────────────────────

    void insert(glm::vec2 pos, std::uint32_t id) {
        insertAt(0, pos, id);
    }

    // ── Accessors ─────────────────────────────────────────────

    [[nodiscard]] const Node& root()     const noexcept { return pool_[0]; }
    [[nodiscard]] const Node& at(int i)  const noexcept { return pool_[i]; }
    [[nodiscard]] std::size_t poolSize() const noexcept { return pool_.size(); }

    // ── Bounding box collection for visualisation ─────────────

    /**
     * Recursively collects bounding boxes of all internal QuadTree cells.
     *
     * Internal cells (non-leaves) represent spatial regions where the
     * Barnes-Hut algorithm aggregated multiple nodes into a super-node.
     * Visualising these boxes reveals the adaptive refinement structure:
     * densely populated areas produce many small cells while sparse
     * regions produce large cells.
     *
     * @param minMass  Minimum node count for a cell to be included.
     *                 Default = 2 suppresses single-node leaf boxes,
     *                 which would clutter the visualisation.
     * @return         Vector of BoundingBox structs, unordered.
     */
    [[nodiscard]] std::vector<BoundingBox>
    collectBoxes(float minMass = 2.0f) const
    {
        std::vector<BoundingBox> result;
        result.reserve(pool_.size() / 2);
        collectRecursive(0, minMass, result);
        return result;
    }

private:
    std::vector<Node> pool_;

    void insertAt(int idx, glm::vec2 pos, std::uint32_t id) {
        Node& n        = pool_[idx];
        n.centerOfMass = (n.centerOfMass * n.totalMass + pos)
                         / (n.totalMass + 1.0f);
        n.totalMass   += 1.0f;

        if (n.isLeaf()) {
            if (!n.hasPoint) {
                n.point    = pos;
                n.pointId  = id;
                n.hasPoint = true;
                return;
            }
            subdivide(idx);
            glm::vec2     oldPt = pool_[idx].point;
            std::uint32_t oldId = pool_[idx].pointId;
            pool_[idx].hasPoint = false;
            routeToChild(idx, oldPt, oldId);
        }
        routeToChild(idx, pos, id);
    }

    void subdivide(int idx) {
        for (int q = 0; q < 4; ++q) {
            pool_[idx].children[q] = static_cast<int>(pool_.size());
            pool_.emplace_back(Node{ pool_[idx].bounds.child(q) });
        }
    }

    void routeToChild(int parentIdx, glm::vec2 pos, std::uint32_t id) {
        int q  = pool_[parentIdx].bounds.quadrant(pos);
        int ci = pool_[parentIdx].children[q];
        if (!pool_[ci].bounds.contains(pos)) {
            for (int qq = 0; qq < 4; ++qq) {
                int alt = pool_[parentIdx].children[qq];
                if (alt != NULL_NODE && pool_[alt].bounds.contains(pos)) {
                    ci = alt;
                    break;
                }
            }
        }
        insertAt(ci, pos, id);
    }

    void collectRecursive(int idx, float minMass,
                          std::vector<BoundingBox>& out) const
    {
        if (idx == NULL_NODE) return;
        const Node& n = pool_[idx];
        if (n.totalMass < minMass) return;

        // Only collect internal nodes — leaf boxes clutter the plot
        if (!n.isLeaf())
            out.push_back(n.bounds);

        for (int q = 0; q < 4; ++q)
            collectRecursive(n.children[q], minMass, out);
    }
};
