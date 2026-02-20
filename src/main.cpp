#include "graph.hpp"
#include "layout_engine.hpp"
#include "exporter.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// ── Simulation parameters ────────────────────────────────────────────────────

struct Config {
    // Graph
    std::size_t  numVertices  = 80;
    double       edgeProb     = 0.08;

    // Frame
    float        frameW       = 1920.0f;
    float        frameH       = 1080.0f;

    // Layout engine
    float        C            = 1.0f;   // k-scaling constant
    float        initTemp     = 200.0f; // T₀ — proportional to frame size
    float        coolingRate  = 0.95f;  // α per iteration

    // Run
    int          maxIter      = 500;

    // I/O
    fs::path     outputDir    = "output";

    // Reproducibility
    std::uint64_t graphSeed   = 42;
    std::uint64_t layoutSeed  = 7;
};

// ── Entry point ──────────────────────────────────────────────────────────────

int main() {
    const Config cfg;

    // ── 1. Build graph ───────────────────────────────────────
    std::cout << "[1/4] Generating Erdős–Rényi G("
              << cfg.numVertices << ", " << cfg.edgeProb << ") ... ";

    Graph g = Graph::erdosRenyi(cfg.numVertices, cfg.edgeProb, cfg.graphSeed);

    std::cout << "done.  |V| = " << g.vertexCount()
              << "  |E| = "      << g.edgeCount() << '\n';

    // ── 2. Initialise layout engine ──────────────────────────
    std::cout << "[2/4] Initialising layout engine ... ";

    LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
    engine.setTemperature(cfg.initTemp);
    engine.setCoolingRate(cfg.coolingRate);
    engine.initialize(g, cfg.layoutSeed);

    std::cout << "done.  k = " << engine.optimalDistance() << '\n';

    // ── 3. Run layout loop ───────────────────────────────────
    std::cout << "[3/4] Running " << cfg.maxIter << " iterations ...\n";

    std::vector<float> convergenceCurve;
    convergenceCurve.reserve(static_cast<std::size_t>(cfg.maxIter));

    for (int iter = 0; iter < cfg.maxIter; ++iter) {
        engine.step(g);
        convergenceCurve.push_back(engine.kineticEnergy());

        // Progress report every 50 iterations
        if ((iter + 1) % 50 == 0)
            std::cout << "  iter " << std::setw(4) << (iter + 1)
                      << "  |  T = "        << std::fixed << std::setprecision(4)
                      << engine.temperature()
                      << "  |  E_k = "      << engine.kineticEnergy() << '\n';
    }

    // ── 4. Export results ────────────────────────────────────
    std::cout << "[4/4] Exporting results to '" << cfg.outputDir << "' ... ";

    try {
        DataExporter::exportAll(g, convergenceCurve, cfg.outputDir);
    } catch (const std::runtime_error& e) {
        std::cerr << "\n[ERROR] " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "done.\n"
              << "  → " << (cfg.outputDir / "nodes.csv")   << '\n'
              << "  → " << (cfg.outputDir / "edges.csv")   << '\n'
              << "  → " << (cfg.outputDir / "metrics.csv") << '\n';

    return EXIT_SUCCESS;
}