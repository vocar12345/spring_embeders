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
 * Header-only utility that serialises graph state, algorithm
 * metrics, and animation frames to CSV files.
 *
 * Static methods handle one-shot exports (nodes, edges, metrics).
 * AnimationWriter is a RAII class for incremental frame appending.
 */
class DataExporter {
public:

    // ── One-shot exports ──────────────────────────────────────

    /**
     * Exports final node positions.
     * Format (nodes.csv):  node_id,x,y
     */
    static void exportNodes(const Graph&    g,
                            const fs::path& outputDir)
    {
        const fs::path path = ensureDir(outputDir) / "nodes.csv";
        std::ofstream  file = openFile(path);

        file << "node_id,x,y\n"
             << std::fixed << std::setprecision(6);

        for (const Node& v : g.nodes())
            file << v.id          << ','
                 << v.position.x  << ','
                 << v.position.y  << '\n';

        checkStream(file, path);
    }

    /**
     * Exports the edge list in canonical (min, max) order.
     * Format (edges.csv):  source,target
     */
    static void exportEdges(const Graph&    g,
                            const fs::path& outputDir)
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
     * Exports the per-iteration kinetic energy convergence curve.
     * Format (metrics.csv):  iteration,kinetic_energy
     */
    static void exportMetrics(std::span<const float> curve,
                              const fs::path&        outputDir)
    {
        const fs::path path = ensureDir(outputDir) / "metrics.csv";
        std::ofstream  file = openFile(path);

        file << "iteration,kinetic_energy\n"
             << std::fixed << std::setprecision(6);

        for (std::size_t i = 0; i < curve.size(); ++i)
            file << i << ',' << curve[i] << '\n';

        checkStream(file, path);
    }

    /// Convenience: exports nodes, edges, and metrics in one call.
    static void exportAll(const Graph&           g,
                          std::span<const float> curve,
                          const fs::path&        outputDir)
    {
        exportNodes  (g, outputDir);
        exportEdges  (g, outputDir);
        exportMetrics(curve, outputDir);
    }

    // ── Animation frame writer (RAII) ─────────────────────────

    /**
     * AnimationWriter
     * ─────────────────────────────────────────────────────────
     * Opens animation_frames.csv and incrementally appends node
     * positions for each sampled iteration.
     *
     * CSV format:
     *   method,iteration,node_id,x,y
     *   BruteForce,0,0,412.31,300.87
     *   BruteForce,0,1,804.12,541.22
     *   ...
     *
     * Usage:
     *   DataExporter::AnimationWriter w("output", true);
     *   for (int i = 0; i < maxIter; ++i) {
     *       engine.step(g);
     *       if (i % 5 == 0)
     *           w.appendFrame(g, "BruteForce", i);
     *   }
     *   // file flushed + closed on destruction
     */
    class AnimationWriter {
    public:
        explicit AnimationWriter(const fs::path& outputDir,
                                 bool            overwrite = true)
        {
            ensureDir(outputDir);
            const fs::path path = outputDir / "animation_frames.csv";

            if (overwrite) {
                file_ = openFile(path);   // truncate existing
            } else {
                file_.open(path, std::ios::app);
                if (!file_.is_open())
                    throw std::runtime_error(
                        "AnimationWriter: cannot open '" +
                        path.string() + "'.");
                return;   // skip header when appending
            }

            file_ << "method,iteration,node_id,x,y\n"
                  << std::fixed << std::setprecision(6);
        }

        /**
         * Appends one frame: all node positions at `iteration`
         * labelled with `methodName`.
         *
         * @param g           Graph after engine.step().
         * @param methodName  Label for the `method` column.
         * @param iteration   0-based iteration index.
         */
        void appendFrame(const Graph&     g,
                         std::string_view methodName,
                         int              iteration)
        {
            for (const Node& v : g.nodes())
                file_ << methodName   << ','
                      << iteration    << ','
                      << v.id         << ','
                      << v.position.x << ','
                      << v.position.y << '\n';
        }

        void flush() { file_.flush(); }

        ~AnimationWriter() {
            if (file_.is_open()) file_.flush();
        }

        // Non-copyable, movable
        AnimationWriter(const AnimationWriter&)            = delete;
        AnimationWriter& operator=(const AnimationWriter&) = delete;
        AnimationWriter(AnimationWriter&&)                 = default;
        AnimationWriter& operator=(AnimationWriter&&)      = default;

    private:
        std::ofstream file_;
    };

private:
    static fs::path ensureDir(const fs::path& dir) {
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec)
            throw std::runtime_error(
                "DataExporter: cannot create directory '" +
                dir.string() + "': " + ec.message());
        return dir;
    }

    static std::ofstream openFile(const fs::path& path) {
        std::ofstream f{ path };
        if (!f.is_open())
            throw std::runtime_error(
                "DataExporter: cannot open '" +
                path.string() + "' for writing.");
        return f;
    }

    static void checkStream(const std::ofstream& f,
                            const fs::path&      path)
    {
        if (!f.good())
            throw std::runtime_error(
                "DataExporter: I/O error writing '" +
                path.string() + "'.");
    }
};
