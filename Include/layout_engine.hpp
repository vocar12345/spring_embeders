#pragma once

#include "graph.hpp"
#include <glm/geometric.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <span>

// ============================================================
//  IRepulsiveStrategy  –  Strategy interface
// ============================================================

/**
 * Any class implementing this interface receives the full node list
 * and is responsible for accumulating repulsive displacement into
 * each node's `displacement` field.
 *
 * Complexity contract:
 *   - BruteForceRepulsion  : O(|V|²)
 *   - BarnesHutRepulsion   : O(|V| log |V|)  [future]
 */
class IRepulsiveStrategy {
public:
    virtual ~IRepulsiveStrategy() = default;

    /**
     * @param nodes  Mutable span over all nodes in the graph.
     * @param k      Optimal distance parameter.
     */
    virtual void computeRepulsive(std::span<Node> nodes, float k) = 0;
};

// ============================================================
//  BruteForceRepulsion  –  O(|V|²) reference implementation
// ============================================================

class BruteForceRepulsion final : public IRepulsiveStrategy {
public:
    void computeRepulsive(std::span<Node> nodes, float k) override {
        const float k2 = k * k;

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            for (std::size_t j = i + 1; j < nodes.size(); ++j) {
                glm::vec2 delta = nodes[i].position - nodes[j].position;
                float dist = glm::length(delta);

                if (dist < 1e-4f) {             // avoid division by zero
                    dist = 1e-4f;
                    delta = glm::vec2{ 1e-4f, 0.0f };
                }

                // f_r(d) = k² / d  →  force vector = (k²/d²) * delta
                glm::vec2 force = (k2 / (dist * dist)) * delta;

                nodes[i].displacement += force;
                nodes[j].displacement -= force;   // Newton's 3rd law
            }
        }
    }
};

// ============================================================
//  LayoutEngine
// ============================================================

class LayoutEngine {
public:
    // ── Construction ─────────────────────────────────────────
    explicit LayoutEngine(float width, float height, float C = 1.0f)
        : W_(width), H_(height), C_(C),
          repulsiveStrategy_(std::make_unique<BruteForceRepulsion>())
    {}

    void setRepulsiveStrategy(std::unique_ptr<IRepulsiveStrategy> s) {
        repulsiveStrategy_ = std::move(s);
    }

    // ── Initialisation ───────────────────────────────────────

    /**
     * Scatters all nodes uniformly at random within the bounding box
     * [0, W] × [0, H] and computes the optimal distance k.
     *
     *   k = C * sqrt(A / |V|),   A = W * H
     */
    void initialize(Graph& g, std::optional<std::uint64_t> seed = std::nullopt) {
        const float A = W_ * H_;
        k_ = C_ * std::sqrt(A / static_cast<float>(g.vertexCount()));

        std::mt19937 rng{ static_cast<std::uint32_t>(seed.value_or(std::random_device{}())) };
        std::uniform_real_distribution<float> rx{ 0.0f, W_ };
        std::uniform_real_distribution<float> ry{ 0.0f, H_ };

        for (Node& v : g.nodes())
            v.position = { rx(rng), ry(rng) };

        lastKineticEnergy_ = 0.0f;
    }

    // ── Cooling schedule ─────────────────────────────────────
    void setTemperature(float t)   noexcept { T_ = t; }
    void setCoolingRate(float r)   noexcept { coolingRate_ = r; }

    // ── Accessors ────────────────────────────────────────────
    [[nodiscard]] float temperature()     const noexcept { return T_; }
    [[nodiscard]] float kineticEnergy()   const noexcept { return lastKineticEnergy_; }
    [[nodiscard]] float optimalDistance() const noexcept { return k_; }

    // ── Core step ────────────────────────────────────────────
    /**
     * Executes one full iteration of the Fruchterman-Reingold algorithm:
     *
     *  1. Reset displacements.
     *  2. Repulsive forces  (delegated to strategy).
     *  3. Attractive forces (along edges only).
     *  4. Clamp displacement to T; apply; boundary-clamp positions.
     *  5. Cool temperature.
     *  6. Record kinetic energy.
     */
    void step(Graph& g) {
        auto& nodes = g.nodes();

        // ── 1. Reset displacements ────────────────────────────
        for (Node& v : nodes) v.resetDisplacement();

        // ── 2. Repulsive forces (strategy-dispatched) ─────────
        repulsiveStrategy_->computeRepulsive(std::span{ nodes }, k_);

        // ── 3. Attractive forces ──────────────────────────────
        // f_a(d) = d² / k  →  force vector = (d / k) * delta_unit
        //                                   = delta * (d / k) / d
        //                                   = delta / k
        for (const Edge& e : g.edges()) {
            Node& u = g.nodeById(e.source);
            Node& v = g.nodeById(e.target);

            glm::vec2 delta = u.position - v.position;
            float dist = glm::length(delta);
            if (dist < 1e-4f) continue;

            // fa(d) = d²/k  ⟹  magnitude, direction = delta/dist
            float mag        = (dist * dist) / k_;
            glm::vec2 force  = (delta / dist) * mag;

            u.displacement -= force;
            v.displacement += force;
        }

        // ── 4. Clamp to T, apply, boundary-clamp ─────────────
        float energy = 0.0f;

        for (Node& v : nodes) {
            float dispLen = glm::length(v.displacement);

            if (dispLen > 1e-6f) {
                float clamped  = std::min(dispLen, T_);
                v.position    += (v.displacement / dispLen) * clamped;
                energy        += clamped;
            }

            // Keep node strictly inside [0,W] × [0,H]
            v.position.x = std::clamp(v.position.x, 0.0f, W_);
            v.position.y = std::clamp(v.position.y, 0.0f, H_);
        }

        lastKineticEnergy_ = energy;

        // ── 5. Simulated annealing: cool temperature ──────────
        T_ = std::max(T_ * coolingRate_, T_min_);
    }

private:
    // Frame parameters
    float W_, H_, C_;

    // Algorithm state
    float k_                { 1.0f  };   // optimal distance
    float T_                { 1.0f  };   // current temperature
    float T_min_            { 1e-3f };   // minimum temperature floor
    float coolingRate_      { 0.95f };   // multiplicative decay α per step

    // Metrics
    float lastKineticEnergy_{ 0.0f };

    // Force strategy (swappable at runtime)
    std::unique_ptr<IRepulsiveStrategy> repulsiveStrategy_;
};