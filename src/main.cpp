#include "graph.hpp"
#include "layout_engine.hpp"
#include "barnes_hut.hpp"
#include "exporter.hpp"
#include <windows.h>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

namespace fs  = std::filesystem;
namespace chr = std::chrono;

// ── Simulation parameters ────────────────────────────────────────────────────

struct Config {
    // Graph  ── larger graph to demonstrate Barnes-Hut benefit
    std::size_t  numVertices  = 1000;
    double       edgeProb     = 0.02;

    // Frame
    float        frameW       = 1920.0f;
    float        frameH       = 1080.0f;

    // Layout engine
    float        C            = 1.0f;
    float        initTemp     = 200.0f;
    float        coolingRate  = 0.95f;

    // Barnes-Hut parameter
    // θ = 0.5 is the standard choice: good accuracy / speed trade-off.
    // Lower θ → more exact (slower); higher θ → more approximate (faster).
    float        theta        = 0.5f;

    // Run
    int          maxIter      = 500;

    // I/O
    fs::path     outputDir    = "output";

    // Reproducibility
    std::uint64_t graphSeed   = 42;
    std::uint64_t layoutSeed  = 7;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Formats a chrono duration as a human-readable string. */
template<typename Duration>
std::string formatDuration(Duration d) {
    using namespace std::chrono;
    auto ms = duration_cast<milliseconds>(d).count();
    if (ms < 1000)
        return std::to_string(ms) + " ms";
    return std::to_string(ms / 1000) + "." +
           std::to_string((ms % 1000) / 10) + " s";
}

// ── Entry point ──────────────────────────────────────────────────────────────

int main() {
    const Config cfg;

    // ── 1. Build graph ───────────────────────────────────────
    std::cout << "[1/4] Generating Erdős–Rényi G("
              << cfg.numVertices << ", " << cfg.edgeProb << ") ... ";
    std::cout.flush();

    Graph g = Graph::erdosRenyi(cfg.numVertices, cfg.edgeProb, cfg.graphSeed);

    std::cout << "done.\n"
              << "       |V| = " << g.vertexCount()
              << "   |E| = "     << g.edgeCount() << '\n';

    // ── 2. Initialise layout engine with Barnes-Hut strategy ─
    std::cout << "[2/4] Initialising LayoutEngine (Barnes-Hut θ = "
              << cfg.theta << ") ... ";
    std::cout.flush();

    LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
    engine.setTemperature(cfg.initTemp);
    engine.setCoolingRate(cfg.coolingRate);

    // Swap the default O(|V|²) strategy for O(|V| log |V|) Barnes-Hut
    engine.setRepulsiveStrategy(
        std::make_unique<BarnesHutRepulsion>(cfg.theta)
    );

    engine.initialize(g, cfg.layoutSeed);

    std::cout << "done.\n"
              << "       k = " << engine.optimalDistance() << '\n';

    // ── 3. Run layout loop — timed with std::chrono ──────────
    std::cout << "[3/4] Running " << cfg.maxIter << " iterations ...\n";

    std::vector<float> convergenceCurve;
    convergenceCurve.reserve(static_cast<std::size_t>(cfg.maxIter));

    // ┌─ Timer start ───────────────────────────────────────────
    const auto timeStart = chr::high_resolution_clock::now();

    for (int iter = 0; iter < cfg.maxIter; ++iter) {
        engine.step(g);
        convergenceCurve.push_back(engine.kineticEnergy());

        if ((iter + 1) % 100 == 0) {
            const auto elapsed = chr::high_resolution_clock::now() - timeStart;
            std::cout << "  iter " << std::setw(4) << (iter + 1)
                      << "  |  T = "    << std::fixed << std::setprecision(4)
                      << std::setw(10)  << engine.temperature()
                      << "  |  E_k = "  << std::setw(12)
                      << engine.kineticEnergy()
                      << "  |  elapsed: " << formatDuration(elapsed)
                      << '\n';
        }
    }

    const auto timeEnd     = chr::high_resolution_clock::now();
    // └─ Timer end ─────────────────────────────────────────────

    const auto totalTime   = timeEnd - timeStart;
    const auto perIterTime = totalTime / cfg.maxIter;

    std::cout << '\n'
              << "  ┌─ Timing summary ─────────────────────────────\n"
              << "  │  Strategy      : Barnes-Hut (θ = " << cfg.theta << ")\n"
              << "  │  |V|           : " << g.vertexCount() << '\n'
              << "  │  Iterations    : " << cfg.maxIter << '\n'
              << "  │  Total time    : " << formatDuration(totalTime) << '\n'
              << "  │  Per iteration : " << formatDuration(perIterTime) << '\n'
              << "  └──────────────────────────────────────────────\n\n";

    // ── 4. Export results ────────────────────────────────────
    std::cout << "[4/4] Exporting results to '" << cfg.outputDir << "' ... ";
    std::cout.flush();

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
