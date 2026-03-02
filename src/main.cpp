#include "graph.hpp"
#include "layout_engine.hpp"
#include "barnes_hut.hpp"
#include "exporter.hpp"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <string>
#include <string_view>
#include <chrono>

namespace fs  = std::filesystem;
namespace chr = std::chrono;

// ── Configuration ─────────────────────────────────────────────────────────────

struct Config {
    // Layout engine
    float frameW      = 1920.0f;
    float frameH      = 1080.0f;
    float C           = 1.0f;
    float initTemp    = 200.0f;
    float coolingRate = 0.95f;
    float theta       = 0.8f;

    // Animation capture
    int   maxIter       = 300;
    int   frameInterval = 5;

    // I/O
    fs::path      outputDir  = "output";
    std::uint64_t graphSeed  = 42;
    std::uint64_t layoutSeed = 7;
};

// ── Topology factory ──────────────────────────────────────────────────────────

/**
 * Builds a Graph based on the topology name from argv[1].
 *
 * Supported topologies:
 *   random     – Erdos-Renyi G(250, 0.04)
 *   grid       – 16x16 lattice grid (256 nodes)
 *   tree       – Balanced binary tree depth=7 (255 nodes)
 *   scale-free – Barabasi-Albert n=250, m0=5, m=3
 *
 * Default (no argument): scale-free
 */
static std::pair<Graph, std::string>
buildGraph(std::string_view topology, std::uint64_t seed)
{
    if (topology == "random") {
        auto g = Graph::erdosRenyi(250, 0.04, seed);
        return { std::move(g), "Erdos-Renyi G(250, 0.04)" };
    }
    if (topology == "grid") {
        auto g = Graph::grid(16, 16);
        return { std::move(g), "Grid 16x16" };
    }
    if (topology == "tree") {
        auto g = Graph::binaryTree(7);   // 2^8 - 1 = 255 nodes
        return { std::move(g), "Binary Tree depth=7" };
    }
    // Default: scale-free
    auto g = Graph::barabasiAlbert(250, 5, 3, seed);
    return { std::move(g), "Barabasi-Albert n=250 m0=5 m=3" };
}

template<typename Duration>
std::string formatMs(Duration d) {
    return std::to_string(
        chr::duration_cast<chr::milliseconds>(d).count()) + " ms";
}

// ── Entry point ───────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const Config cfg;

    // Parse topology from command-line argument
    std::string_view topology = (argc > 1) ? argv[1] : "scale-free";

    std::cout << "Topology: " << topology << "\n\n";

    // ── 1. Build graph ────────────────────────────────────────
    std::cout << "[1/6] Building graph ... ";
    std::cout.flush();

    auto [gBF, description] = buildGraph(topology, cfg.graphSeed);

    std::cout << "done.\n"
              << "       " << description << "\n"
              << "       |V| = " << gBF.vertexCount()
              << "   |E| = "     << gBF.edgeCount() << '\n';

    // ── 2. Deep-copy for Barnes-Hut run ───────────────────────
    std::cout << "[2/6] Deep-copying graph ... ";
    Graph gBH = gBF;
    std::cout << "done.\n";

    // ── 3. Export edge list ───────────────────────────────────
    std::cout << "[3/6] Exporting edges ... ";
    DataExporter::exportEdges(gBF, cfg.outputDir);
    std::cout << "done.\n";

    DataExporter::AnimationWriter animWriter{ cfg.outputDir, true };

    // ── 4. Brute-Force run ────────────────────────────────────
    std::cout << "[4/6] BruteForce run (" << cfg.maxIter << " iters) ...\n";
    {
        LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
        engine.setTemperature(cfg.initTemp);
        engine.setCoolingRate(cfg.coolingRate);
        engine.initialize(gBF, cfg.layoutSeed);

        animWriter.appendFrame(gBF, "BruteForce", 0);

        const auto t0 = chr::high_resolution_clock::now();
        for (int i = 1; i <= cfg.maxIter; ++i) {
            engine.step(gBF);
            if (i % cfg.frameInterval == 0)
                animWriter.appendFrame(gBF, "BruteForce", i);
        }
        std::cout << "  Done in "
                  << formatMs(chr::high_resolution_clock::now() - t0) << '\n';
    }

    // ── 5. Barnes-Hut run ─────────────────────────────────────
    std::cout << "[5/6] BarnesHut run (theta=" << cfg.theta
              << ", " << cfg.maxIter << " iters) ...\n";
    {
        LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
        engine.setTemperature(cfg.initTemp);
        engine.setCoolingRate(cfg.coolingRate);
        engine.setRepulsiveStrategy(
            std::make_unique<BarnesHutRepulsion>(cfg.theta));
        engine.initialize(gBH, cfg.layoutSeed);

        animWriter.appendFrame(gBH, "BarnesHut", 0);

        const auto t0 = chr::high_resolution_clock::now();
        for (int i = 1; i <= cfg.maxIter; ++i) {
            engine.step(gBH);
            if (i % cfg.frameInterval == 0)
                animWriter.appendFrame(gBH, "BarnesHut", i);
        }
        std::cout << "  Done in "
                  << formatMs(chr::high_resolution_clock::now() - t0) << '\n';
    }

    animWriter.flush();

    // ── 6. Build final QuadTree on converged BF positions ─────
    // We use BruteForce final positions as the "ground truth" layout
    // since it is exact. The QuadTree export visualises how Barnes-Hut
    // would subdivide this converged configuration.
    std::cout << "[6/6] Building final QuadTree and exporting ...\n";
    {
        // Compute tight bounding box around converged positions
        float minX =  std::numeric_limits<float>::max();
        float minY =  std::numeric_limits<float>::max();
        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        for (const Node& v : gBF.nodes()) {
            minX = std::min(minX, v.position.x);
            minY = std::min(minY, v.position.y);
            maxX = std::max(maxX, v.position.x);
            maxY = std::max(maxY, v.position.y);
        }
        const float margin = 10.0f;
        BoundingBox rootBox{
            { (minX + maxX) * 0.5f, (minY + maxY) * 0.5f },
            (maxX - minX) * 0.5f + margin,
            (maxY - minY) * 0.5f + margin
        };

        QuadTree finalTree{ rootBox, gBF.vertexCount() };
        for (const Node& v : gBF.nodes())
            finalTree.insert(v.position, v.id);

        // Export node positions and QuadTree cells
        DataExporter::exportNodes(gBF, cfg.outputDir);
        DataExporter::exportQuadTree(finalTree, cfg.outputDir);
    }

    std::cout << "\nAll outputs in '" << cfg.outputDir << "/':\n"
              << "  edges.csv  nodes.csv  animation_frames.csv\n"
              << "  quadtree.csv  metrics not exported in animation mode\n\n"
              << "Next steps:\n"
              << "  python animate.py\n"
              << "  python plot_quadtree.py\n";

    return EXIT_SUCCESS;
}
