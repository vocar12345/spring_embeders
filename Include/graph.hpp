#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include <ranges>
#include <concepts>
#include <cstdint>

// ============================================================
//  Node
// ============================================================

struct Node {
    using ID = std::uint32_t;

    ID         id;
    glm::vec2  position    { 0.0f, 0.0f };
    glm::vec2  displacement{ 0.0f, 0.0f };

    explicit Node(ID id) : id(id) {}

    /// Resets displacement accumulator at the start of each iteration.
    void resetDisplacement() noexcept { displacement = glm::vec2{ 0.0f }; }
};

// ============================================================
//  Edge
// ============================================================

struct Edge {
    Node::ID source;
    Node::ID target;

    Edge(Node::ID u, Node::ID v) : source(u), target(v) {}

    /// Canonical form: smaller ID first – ensures undirected uniqueness.
    [[nodiscard]] Edge canonical() const noexcept {
        return (source <= target) ? *this : Edge{ target, source };
    }

    bool operator==(const Edge& o) const noexcept {
        auto a = canonical(), b = o.canonical();
        return a.source == b.source && a.target == b.target;
    }
};

struct EdgeHash {
    std::size_t operator()(const Edge& e) const noexcept {
        auto ce = e.canonical();
        // Szudzik pairing for two 32-bit IDs
        std::size_t a = ce.source, b = ce.target;
        return (a >= b) ? a * a + a + b : a + b * b;
    }
};

// ============================================================
//  Graph
// ============================================================

class Graph {
public:
    // ── Accessors ────────────────────────────────────────────
    [[nodiscard]] std::size_t vertexCount() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t edgeCount()   const noexcept { return edges_.size(); }

    [[nodiscard]] const std::vector<Node>& nodes() const noexcept { return nodes_; }
    [[nodiscard]]       std::vector<Node>& nodes()       noexcept { return nodes_; }

    [[nodiscard]] const std::unordered_set<Edge, EdgeHash>& edges() const noexcept {
        return edges_;
    }

    /// Returns the adjacency list for node `id` (neighbour IDs).
    [[nodiscard]] const std::vector<Node::ID>& neighbours(Node::ID id) const {
        return adjacency_.at(id);
    }

    // ── Mutation ─────────────────────────────────────────────

    /// Adds a vertex and returns a reference to it.
    Node& addVertex(Node::ID id) {
        if (index_.contains(id))
            throw std::invalid_argument("Vertex already exists.");
        index_[id] = nodes_.size();
        adjacency_[id] = {};
        return nodes_.emplace_back(id);
    }

    /// Adds an undirected edge (u, v). Both endpoints must already exist.
    void addEdge(Node::ID u, Node::ID v) {
        requireVertex(u); requireVertex(v);
        Edge e{ u, v };
        if (edges_.insert(e).second) {          // inserted – not a duplicate
            adjacency_[u].push_back(v);
            adjacency_[v].push_back(u);         // undirected: symmetric lists
        }
    }

    [[nodiscard]] Node& nodeById(Node::ID id) {
        return nodes_[index_.at(id)];
    }
    [[nodiscard]] const Node& nodeById(Node::ID id) const {
        return nodes_[index_.at(id)];
    }

    // ── Erdős–Rényi G(n, p) generator ────────────────────────
    /**
     * Generates a random graph with `n` vertices where each possible
     * undirected edge exists independently with probability `p`.
     *
     * @param n  Number of vertices.
     * @param p  Edge probability ∈ [0, 1].
     * @param seed  RNG seed (defaults to random_device).
     */
    static Graph erdosRenyi(std::size_t n, double p,
                            std::optional<std::uint64_t> seed = std::nullopt)
    {
        if (p < 0.0 || p > 1.0)
            throw std::domain_error("Edge probability p must be in [0, 1].");

        Graph g;
        for (std::size_t i = 0; i < n; ++i)
            g.addVertex(static_cast<Node::ID>(i));

        std::mt19937_64 rng{ static_cast<std::uint64_t>(seed.value_or(std::random_device{}())) };
        std::bernoulli_distribution coin{ p };

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                if (coin(rng))
                    g.addEdge(static_cast<Node::ID>(i),
                              static_cast<Node::ID>(j));
        return g;
    }

private:
    std::vector<Node>                          nodes_;
    std::unordered_set<Edge, EdgeHash>         edges_;
    std::unordered_map<Node::ID, std::size_t>  index_;       // id → nodes_ index
    std::unordered_map<Node::ID, std::vector<Node::ID>> adjacency_;

    void requireVertex(Node::ID id) const {
        if (!index_.contains(id))
            throw std::invalid_argument("Vertex does not exist.");
    }
};