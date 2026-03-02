#include "graph.hpp"
#include "layout_engine.hpp"
#include "barnes_hut.hpp"
#include "exporter.hpp"

#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

namespace fs  = std::filesystem;
namespace chr = std::chrono;

struct Config {
    // Graph — V=250 keeps animation visually clean
    std::size_t   numVertices   = 250;
    double        edgeProb      = 0.04;   // avg degree ~10

    // Frame
    float         frameW        = 1920.0f;
    float         frameH        = 1080.0f;
    float         C             = 1.0f;

    // Layout
    float         initTemp      = 200.0f;
    float         coolingRate   = 0.95f;

    // Barnes-Hut: theta=0.8 gives better spread across the frame
    // (higher theta = more aggressive long-range repulsion approximation
    // = nodes pushed further apart, visually closer to brute force result)
    float         theta         = 0.8f;

    // Simulation
    int           maxIter       = 300;
    int           frameInterval = 5;      // capture every Nth iteration

    // I/O
    fs::path      outputDir     = "output";

    // Same seeds guarantee identical initial positions for both methods
    std::uint64_t graphSeed     = 42;
    std::uint64_t layoutSeed    = 7;
};

template<typename Duration>
std::string formatMs(Duration d) {
    auto ms = chr::duration_cast<chr::milliseconds>(d).count();
    return std::to_string(ms) + " ms";
}

int main() {
    const Config cfg;

    // ── 1. Build graph ────────────────────────────────────────
    std::cout << "[1/5] Generating Erdos-Renyi G("
              << cfg.numVertices << ", " << cfg.edgeProb << ") ... ";
    std::cout.flush();

    Graph gBF = Graph::erdosRenyi(cfg.numVertices, cfg.edgeProb, cfg.graphSeed);

    std::cout << "done.  |V| = " << gBF.vertexCount()
              << "  |E| = "      << gBF.edgeCount() << '\n';

    // ── 2. Deep-copy graph ────────────────────────────────────
    // Graph uses STL containers so assignment is a full deep copy.
    // Both engines use the same layoutSeed so initial positions are
    // identical — making the side-by-side comparison scientifically fair.
    std::cout << "[2/5] Deep-copying graph for Barnes-Hut run ... ";
    Graph gBH = gBF;
    std::cout << "done.\n";

    // ── 3. Export shared edge list ────────────────────────────
    std::cout << "[3/5] Exporting edges ... ";
    DataExporter::exportEdges(gBF, cfg.outputDir);
    std::cout << "done.\n";

    // Open animation CSV — writes header, truncates any existing file
    DataExporter::AnimationWriter animWriter{ cfg.outputDir, /*overwrite=*/true };

    // ── 4. Brute-Force run ────────────────────────────────────
    std::cout << "[4/5] BruteForce run (" << cfg.maxIter << " iters) ...\n";
    {
        LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
        engine.setTemperature(cfg.initTemp);
        engine.setCoolingRate(cfg.coolingRate);
        // Default strategy is BruteForceRepulsion — no swap needed
        engine.initialize(gBF, cfg.layoutSeed);

        // Capture iteration 0 (initial random scatter)
        animWriter.appendFrame(gBF, "BruteForce", 0);

        const auto t0 = chr::high_resolution_clock::now();

        for (int i = 1; i <= cfg.maxIter; ++i) {
            engine.step(gBF);
            if (i % cfg.frameInterval == 0)
                animWriter.appendFrame(gBF, "BruteForce", i);
            if (i % 100 == 0)
                std::cout << "  iter " << std::setw(4) << i
                          << "  T="  << std::fixed << std::setprecision(3)
                          << engine.temperature()
                          << "  E_k=" << engine.kineticEnergy() << '\n';
        }

        std::cout << "  Done in "
                  << formatMs(chr::high_resolution_clock::now() - t0) << '\n';
    }

    // ── 5. Barnes-Hut run ─────────────────────────────────────
    std::cout << "[5/5] BarnesHut run (theta=" << cfg.theta
              << ", " << cfg.maxIter << " iters) ...\n";
    {
        LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
        engine.setTemperature(cfg.initTemp);
        engine.setCoolingRate(cfg.coolingRate);
        engine.setRepulsiveStrategy(
            std::make_unique<BarnesHutRepulsion>(cfg.theta));
        engine.initialize(gBH, cfg.layoutSeed);

        // Capture iteration 0
        animWriter.appendFrame(gBH, "BarnesHut", 0);

        const auto t0 = chr::high_resolution_clock::now();

        for (int i = 1; i <= cfg.maxIter; ++i) {
            engine.step(gBH);
            if (i % cfg.frameInterval == 0)
                animWriter.appendFrame(gBH, "BarnesHut", i);
            if (i % 100 == 0)
                std::cout << "  iter " << std::setw(4) << i
                          << "  T="  << std::fixed << std::setprecision(3)
                          << engine.temperature()
                          << "  E_k=" << engine.kineticEnergy() << '\n';
        }

        std::cout << "  Done in  "
                  << formatMs(chr::high_resolution_clock::now() - t0) << '\n';
    }

    animWriter.flush();

    const int totalFrames = cfg.maxIter / cfg.frameInterval + 1;
    std::cout << "\nOutput:\n"
              << "  -> " << (cfg.outputDir / "edges.csv") << '\n'
              << "  -> " << (cfg.outputDir / "animation_frames.csv")
              << "  (" << totalFrames * 2 << " total frames, "
              << cfg.numVertices << " nodes each)\n\n"
              << "Next:  python animate.py\n";

    return EXIT_SUCCESS;
}
