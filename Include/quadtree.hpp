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
        if  (right  &&  up) return 0; // NE
        if  (!right &&  up) return 1; // NW
        if  (!right && !up) return 2; // SW
        return 3;                     // SE
    }
};

// ============================================================
//  QuadTree  –  flat pool allocator, zero per-iteration heap
// ============================================================

/**
 * Pool-based QuadTree: all nodes live in a pre-allocated std::vector.
 * No dynamic allocation per insert — avoids the cache-miss penalty of
 * unique_ptr chains that makes pointer-based trees slower than brute
 * force at small-to-medium N.
 *
 * Pool index NULL_NODE (-1) means "no child".
 *
 * Relation to optimal distance k:
 *   The acceptance criterion s/d < θ compares the cell's spatial size s
 *   to the distance d from query node to cell centre of mass.
 *   k sets the repulsion length scale; s and d share the same spatial
 *   units so the criterion is scale-invariant and independent of k.
 */
class QuadTree {
public:
    static constexpr int NULL_NODE = -1;

    struct Node {
        BoundingBox   bounds;
        glm::vec2     centerOfMass{ 0.0f, 0.0f };
        float         totalMass   { 0.0f };
        glm::vec2     point       { 0.0f, 0.0f }; // leaf payload
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
        pool_.push_back(Node{ bounds });  // index 0 = root
    }

    /// Clears the tree and resets with a new bounding box.
    /// Reuses already-allocated pool memory — no heap activity.
    void reset(BoundingBox bounds) {
        pool_.clear();
        pool_.push_back(Node{ bounds });
    }

    // ── Insertion ─────────────────────────────────────────────
    void insert(glm::vec2 pos, std::uint32_t id) {
        insertAt(0, pos, id);
    }

    // ── Accessors ────────────────────────────────────────────
    [[nodiscard]] const Node& root()        const noexcept { return pool_[0];   }
    [[nodiscard]] const Node& at(int i)     const noexcept { return pool_[i];   }

private:
    std::vector<Node> pool_;

    void insertAt(int idx, glm::vec2 pos, std::uint32_t id) {
        // Update centre of mass via online weighted mean
        Node& n       = pool_[idx];
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
            // Occupied leaf: subdivide then push existing point down
            subdivide(idx);
            glm::vec2    oldPt  = pool_[idx].point;
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
            // NOTE: emplace_back may reallocate pool_, so we re-read
            // pool_[idx] implicitly through pool_ on next iteration.
        }
    }

    void routeToChild(int parentIdx, glm::vec2 pos, std::uint32_t id) {
        int q = pool_[parentIdx].bounds.quadrant(pos);
        // Boundary guard: if float rounding puts pos in the wrong child,
        // scan all four children for the one that actually contains it.
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
};
