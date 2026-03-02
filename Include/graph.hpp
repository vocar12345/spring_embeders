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
#include <optional>

// ============================================================
//  Node
// ============================================================

struct Node {
    using ID = std::uint32_t;

    ID        id;
    glm::vec2 position    { 0.0f, 0.0f };
    glm::vec2 displacement{ 0.0f, 0.0f };

    explicit Node(ID id) : id(id) {}

    void resetDisplacement() noexcept { displacement = glm::vec2{ 0.0f }; }
};

// ============================================================
//  Edge
// ============================================================

struct Edge {
    Node::ID source;
    Node::ID target;

    Edge(Node::ID u, Node::ID v) : source(u), target(v) {}

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
        std::size_t a = ce.source, b = ce.target;
        return (a >= b) ? a * a + a + b : a + b * b;
    }
};

// ============================================================
//  Graph
// ============================================================

class Graph {
public:
    // ── Accessors ─────────────────────────────────────────────
    [[nodiscard]] std::size_t vertexCount() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t edgeCount()   const noexcept { return edges_.size(); }

    [[nodiscard]] const std::vector<Node>& nodes() const noexcept { return nodes_; }
    [[nodiscard]]       std::vector<Node>& nodes()       noexcept { return nodes_; }

    [[nodiscard]] const std::unordered_set<Edge, EdgeHash>& edges() const noexcept {
        return edges_;
    }

    [[nodiscard]] const std::vector<Node::ID>& neighbours(Node::ID id) const {
        return adjacency_.at(id);
    }

    // ── Mutation ──────────────────────────────────────────────

    Node& addVertex(Node::ID id) {
        if (index_.contains(id))
            throw std::invalid_argument("Vertex already exists.");
        index_[id] = nodes_.size();
        adjacency_[id] = {};
        return nodes_.emplace_back(id);
    }

    void addEdge(Node::ID u, Node::ID v) {
        requireVertex(u); requireVertex(v);
        Edge e{ u, v };
        if (edges_.insert(e).second) {
            adjacency_[u].push_back(v);
            adjacency_[v].push_back(u);
        }
    }

    [[nodiscard]] Node& nodeById(Node::ID id) {
        return nodes_[index_.at(id)];
    }
    [[nodiscard]] const Node& nodeById(Node::ID id) const {
        return nodes_[index_.at(id)];
    }

    // ── Generators ────────────────────────────────────────────

    /**
     * Erdős–Rényi G(n, p): each edge exists independently with probability p.
     */
    static Graph erdosRenyi(std::size_t n, double p,
                            std::optional<std::uint64_t> seed = std::nullopt)
    {
        if (p < 0.0 || p > 1.0)
            throw std::domain_error("Edge probability p must be in [0, 1].");

        Graph g;
        for (std::size_t i = 0; i < n; ++i)
            g.addVertex(static_cast<Node::ID>(i));

        std::mt19937_64 rng{ seed.value_or(std::random_device{}()) };
        std::bernoulli_distribution coin{ p };

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                if (coin(rng))
                    g.addEdge(static_cast<Node::ID>(i),
                              static_cast<Node::ID>(j));
        return g;
    }

    /**
     * 2D lattice grid of width x height nodes.
     * Each node (r, c) connects to its right and bottom neighbours.
     * Node ID: id = r * width + c
     */
    static Graph grid(std::size_t width, std::size_t height) {
        if (width == 0 || height == 0)
            throw std::domain_error("Grid dimensions must be > 0.");

        Graph g;
        for (std::size_t r = 0; r < height; ++r)
            for (std::size_t c = 0; c < width; ++c)
                g.addVertex(static_cast<Node::ID>(r * width + c));

        // Horizontal edges
        for (std::size_t r = 0; r < height; ++r)
            for (std::size_t c = 0; c + 1 < width; ++c)
                g.addEdge(static_cast<Node::ID>(r * width + c),
                          static_cast<Node::ID>(r * width + c + 1));

        // Vertical edges
        for (std::size_t r = 0; r + 1 < height; ++r)
            for (std::size_t c = 0; c < width; ++c)
                g.addEdge(static_cast<Node::ID>(r       * width + c),
                          static_cast<Node::ID>((r + 1) * width + c));
        return g;
    }

    /**
     * Perfectly balanced binary tree of given depth.
     * Depth 0 = single root node.
     * Total nodes = 2^(depth+1) - 1.
     * Node IDs follow BFS order: root=0, children of i are 2i+1 and 2i+2.
     */
    static Graph binaryTree(std::size_t depth) {
        const std::size_t n = (std::size_t(1) << (depth + 1)) - 1;

        Graph g;
        for (std::size_t i = 0; i < n; ++i)
            g.addVertex(static_cast<Node::ID>(i));

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t left  = 2 * i + 1;
            const std::size_t right = 2 * i + 2;
            if (left  < n) g.addEdge(static_cast<Node::ID>(i),
                                     static_cast<Node::ID>(left));
            if (right < n) g.addEdge(static_cast<Node::ID>(i),
                                     static_cast<Node::ID>(right));
        }
        return g;
    }

    /**
     * Barabasi-Albert scale-free network via preferential attachment.
     *
     * Algorithm:
     *   1. Start with a fully-connected seed of m0 nodes.
     *   2. Add nodes one at a time up to total n.
     *   3. Each new node attaches to m existing nodes chosen with
     *      probability proportional to their current degree
     *      ("rich get richer" / preferential attachment).
     *
     * Produces power-law degree distribution P(k) ~ k^(-3),
     * characteristic of real-world networks.
     *
     * @param n     Total number of nodes.
     * @param m0    Size of the initial fully-connected seed (>= m).
     * @param m     Edges each new node adds (<= m0).
     * @param seed  RNG seed for reproducibility.
     */
    static Graph barabasiAlbert(std::size_t n,
                                std::size_t m0,
                                std::size_t m,
                                std::optional<std::uint64_t> seed = std::nullopt)
    {
        if (m0 < 1 || m < 1)
            throw std::domain_error("BA: m0 and m must be >= 1.");
        if (m > m0)
            throw std::domain_error("BA: m must be <= m0.");
        if (n < m0)
            throw std::domain_error("BA: n must be >= m0.");

        std::mt19937_64 rng{ seed.value_or(std::random_device{}()) };

        Graph g;

        // ── Seed: fully connected subgraph ────────────────────
        for (std::size_t i = 0; i < m0; ++i)
            g.addVertex(static_cast<Node::ID>(i));
        for (std::size_t i = 0; i < m0; ++i)
            for (std::size_t j = i + 1; j < m0; ++j)
                g.addEdge(static_cast<Node::ID>(i),
                          static_cast<Node::ID>(j));

        // Degree list: each node appears degree(v) times.
        // Uniform draw ≡ sampling proportional to degree.
        std::vector<Node::ID> degreeList;
        degreeList.reserve(n * m * 2);
        for (std::size_t i = 0; i < m0; ++i) {
            const auto deg = g.adjacency_.at(static_cast<Node::ID>(i)).size();
            for (std::size_t k = 0; k < deg; ++k)
                degreeList.push_back(static_cast<Node::ID>(i));
        }

        // ── Preferential attachment ───────────────────────────
        for (std::size_t newId = m0; newId < n; ++newId) {
            g.addVertex(static_cast<Node::ID>(newId));

            std::unordered_set<Node::ID> chosen;
            chosen.reserve(m);

            while (chosen.size() < m) {
                std::uniform_int_distribution<std::size_t>
                    dist{ 0, degreeList.size() - 1 };
                Node::ID target = degreeList[dist(rng)];
                if (target != static_cast<Node::ID>(newId))
                    chosen.insert(target);
            }

            for (Node::ID target : chosen) {
                g.addEdge(static_cast<Node::ID>(newId), target);
                degreeList.push_back(static_cast<Node::ID>(newId));
                degreeList.push_back(target);
            }
        }

        return g;
    }

private:
    std::vector<Node>                                    nodes_;
    std::unordered_set<Edge, EdgeHash>                   edges_;
    std::unordered_map<Node::ID, std::size_t>            index_;
    std::unordered_map<Node::ID, std::vector<Node::ID>>  adjacency_;

    void requireVertex(Node::ID id) const {
        if (!index_.contains(id))
            throw std::invalid_argument("Vertex does not exist.");
    }
};
