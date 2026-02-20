#pragma once

#include "graph.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

/**
 * DataExporter
 * ─────────────────────────────────────────────────────────────
 * Header-only utility that serialises graph state and algorithm
 * metrics to CSV files for downstream analysis (Python / R / etc.).
 *
 * All methods are static; no instance state is required.
 * Every method throws std::runtime_error on I/O failure so the
 * caller can decide how to handle it.
 */
class DataExporter {
public:
    // ── Public API ────────────────────────────────────────────

    /**
     * Exports final node positions.
     *
     * Output format (nodes.csv):
     *   node_id,x,y
     *   0,412.31,300.87
     *   ...
     *
     * @param g          The graph after layout has converged.
     * @param outputDir  Directory in which to create the file.
     */
    static void exportNodes(const Graph&     g,
                            const fs::path&  outputDir)
    {
        const fs::path path = ensureDir(outputDir) / "nodes.csv";
        std::ofstream  file = openFile(path);

        file << "node_id,x,y\n";
        file << std::fixed << std::setprecision(6);

        for (const Node& v : g.nodes())
            file << v.id           << ','
                 << v.position.x   << ','
                 << v.position.y   << '\n';

        checkStream(file, path);
    }

    /**
     * Exports the edge list.
     *
     * Output format (edges.csv):
     *   source,target
     *   0,5
     *   ...
     *
     * Each undirected edge is written once in canonical (min, max) order.
     *
     * @param g          The graph.
     * @param outputDir  Directory in which to create the file.
     */
    static void exportEdges(const Graph&     g,
                            const fs::path&  outputDir)
    {
        const fs::path path = ensureDir(outputDir) / "edges.csv";
        std::ofstream  file = openFile(path);

        file << "source,target\n";

        for (const Edge& e : g.edges()) {
            const auto ce = e.canonical();
            file << ce.source << ',' << ce.target << '\n';
        }

        checkStream(file, path);
    }

    /**
     * Exports the per-iteration kinetic energy (convergence curve).
     *
     * Output format (metrics.csv):
     *   iteration,kinetic_energy
     *   0,4821.34
     *   1,4102.87
     *   ...
     *
     * @param curve      Vector of kinetic energy values; index == iteration.
     * @param outputDir  Directory in which to create the file.
     */
    static void exportMetrics(std::span<const float> curve,
                              const fs::path&        outputDir)
    {
        const fs::path path = ensureDir(outputDir) / "metrics.csv";
        std::ofstream  file = openFile(path);

        file << "iteration,kinetic_energy\n";
        file << std::fixed << std::setprecision(6);

        for (std::size_t i = 0; i < curve.size(); ++i)
            file << i << ',' << curve[i] << '\n';

        checkStream(file, path);
    }

    /**
     * Convenience overload: exports all three files in one call.
     */
    static void exportAll(const Graph&           g,
                          std::span<const float> curve,
                          const fs::path&        outputDir)
    {
        exportNodes  (g, outputDir);
        exportEdges  (g, outputDir);
        exportMetrics(curve, outputDir);
    }

private:
    // ── Helpers ───────────────────────────────────────────────

    /// Creates the directory (and any parents) if it does not yet exist.
    static fs::path ensureDir(const fs::path& dir) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec)
            throw std::runtime_error("DataExporter: cannot create directory '"
                                     + dir.string() + "': " + ec.message());
        return dir;
    }

    /// Opens a file for writing; throws on failure.
    static std::ofstream openFile(const fs::path& path) {
        std::ofstream f{ path };
        if (!f.is_open())
            throw std::runtime_error("DataExporter: cannot open '"
                                     + path.string() + "' for writing.");
        return f;
    }

    /// Verifies the stream is still healthy after all writes.
    static void checkStream(const std::ofstream& f, const fs::path& path) {
        if (!f.good())
            throw std::runtime_error("DataExporter: I/O error while writing '"
                                     + path.string() + "'.");
    }
};