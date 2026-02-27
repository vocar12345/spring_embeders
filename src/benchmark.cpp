/**
 * benchmark.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Empirical complexity benchmark: BruteForce O(|V|²) vs Barnes-Hut O(|V|logN).
 *
 * N range extends to 5000 so that Barnes-Hut's crossover advantage is
 * clearly visible on the resulting complexity plots.
 *
 * Output: output/benchmark.csv
 *   Columns: N, BruteForce_ms, BarnesHut_ms
 */

#include "graph.hpp"
#include "layout_engine.hpp"
#include "barnes_hut.hpp"
#include "exporter.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace fs  = std::filesystem;
namespace chr = std::chrono;

// ── Configuration ─────────────────────────────────────────────────────────────

struct BenchConfig {
    // Extended range: crossover between BF and BH typically at N ~ 3000-5000
    std::vector<std::size_t> vertexCounts{
        100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000
    };

    // p = targetDegree / N  →  sparse, realistic graphs
    double targetDegree = 5.0;

    float frameW      = 1920.0f;
    float frameH      = 1080.0f;
    float C           = 1.0f;
    float initTemp    = 200.0f;
    float coolingRate = 0.95f;
    float theta       = 0.5f;

    // 50 iterations: enough to measure the repulsion loop hot path
    // without making large-N brute-force runs take hours.
    int iterations = 50;

    std::uint64_t graphSeed  = 42;
    std::uint64_t layoutSeed = 7;

    fs::path outputDir = "output";
};

// ── Timing helper ─────────────────────────────────────────────────────────────

static double measureMs(LayoutEngine&      engine,
                        Graph&             g,
                        const BenchConfig& cfg)
{
    engine.initialize(g, cfg.layoutSeed);

    const auto t0 = chr::high_resolution_clock::now();
    for (int i = 0; i < cfg.iterations; ++i)
        engine.step(g);
    const auto t1 = chr::high_resolution_clock::now();

    return static_cast<double>(
        chr::duration_cast<chr::microseconds>(t1 - t0).count()) / 1000.0;
}

// ── Result record ─────────────────────────────────────────────────────────────

struct BenchResult {
    std::size_t N;
    double      bruteForceMs;
    double      barnesHutMs;
};

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    const BenchConfig cfg;

    std::cout << "Fruchterman-Reingold Complexity Benchmark\n"
              << "==========================================\n"
              << "Iterations per run : " << cfg.iterations    << '\n'
              << "Barnes-Hut theta   : " << cfg.theta         << '\n'
              << "Target avg degree  : " << cfg.targetDegree  << "\n\n"
              << std::left
              << std::setw(8)  << "N"
              << std::setw(20) << "BruteForce (ms)"
              << std::setw(20) << "BarnesHut (ms)"
              << "Speedup\n"
              << std::string(60, '-') << '\n';

    std::vector<BenchResult> results;
    results.reserve(cfg.vertexCounts.size());

    for (std::size_t N : cfg.vertexCounts) {
        const double p = std::min(cfg.targetDegree / static_cast<double>(N),
                                  1.0);
        Graph g = Graph::erdosRenyi(N, p, cfg.graphSeed);

        // ── Brute force ───────────────────────────────────────
        LayoutEngine bfEngine{ cfg.frameW, cfg.frameH, cfg.C };
        bfEngine.setTemperature(cfg.initTemp);
        bfEngine.setCoolingRate(cfg.coolingRate);
        const double bfMs = measureMs(bfEngine, g, cfg);

        // ── Barnes-Hut ────────────────────────────────────────
        LayoutEngine bhEngine{ cfg.frameW, cfg.frameH, cfg.C };
        bhEngine.setTemperature(cfg.initTemp);
        bhEngine.setCoolingRate(cfg.coolingRate);
        bhEngine.setRepulsiveStrategy(
            std::make_unique<BarnesHutRepulsion>(cfg.theta));
        const double bhMs = measureMs(bhEngine, g, cfg);

        const double speedup = (bhMs > 0.0) ? bfMs / bhMs : 0.0;
        results.push_back({ N, bfMs, bhMs });

        std::cout << std::left  << std::fixed << std::setprecision(2)
                  << std::setw(8)  << N
                  << std::setw(20) << bfMs
                  << std::setw(20) << bhMs
                  << std::setprecision(1) << speedup << "x\n";
    }

    // ── Export CSV ────────────────────────────────────────────
    fs::create_directories(cfg.outputDir);
    const fs::path csvPath = cfg.outputDir / "benchmark.csv";
    std::ofstream  csv{ csvPath };

    if (!csv.is_open()) {
        std::cerr << "[ERROR] Cannot open " << csvPath << '\n';
        return EXIT_FAILURE;
    }

    csv << "N,BruteForce_ms,BarnesHut_ms\n"
        << std::fixed << std::setprecision(4);

    for (const auto& r : results)
        csv << r.N            << ','
            << r.bruteForceMs << ','
            << r.barnesHutMs  << '\n';

    std::cout << "\nResults saved to: " << csvPath << '\n';
    return EXIT_SUCCESS;
}
