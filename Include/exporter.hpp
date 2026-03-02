#pragma once

#include "graph.hpp"
#include "quadtree.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

class DataExporter {
public:

    // ── Static exports ────────────────────────────────────────

    static void exportNodes(const Graph& g, const fs::path& outputDir) {
        const fs::path path = ensureDir(outputDir) / "nodes.csv";
        std::ofstream  file = openFile(path);
        file << "node_id,x,y\n" << std::fixed << std::setprecision(6);
        for (const Node& v : g.nodes())
            file << v.id << ',' << v.position.x << ',' << v.position.y << '\n';
        checkStream(file, path);
    }

    static void exportEdges(const Graph& g, const fs::path& outputDir) {
        const fs::path path = ensureDir(outputDir) / "edges.csv";
        std::ofstream  file = openFile(path);
        file << "source,target\n";
        for (const Edge& e : g.edges()) {
            const auto ce = e.canonical();
            file << ce.source << ',' << ce.target << '\n';
        }
        checkStream(file, path);
    }

    static void exportMetrics(std::span<const float> curve,
                              const fs::path& outputDir) {
        const fs::path path = ensureDir(outputDir) / "metrics.csv";
        std::ofstream  file = openFile(path);
        file << "iteration,kinetic_energy\n" << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < curve.size(); ++i)
            file << i << ',' << curve[i] << '\n';
        checkStream(file, path);
    }

    static void exportAll(const Graph& g, std::span<const float> curve,
                          const fs::path& outputDir) {
        exportNodes(g, outputDir);
        exportEdges(g, outputDir);
        exportMetrics(curve, outputDir);
    }

    /**
     * Exports QuadTree bounding boxes to quadtree.csv.
     *
     * Collects all internal cell bounding boxes from the QuadTree
     * built on the final converged node positions. These cells
     * reveal the adaptive spatial subdivision structure of the
     * Barnes-Hut algorithm — densely populated regions produce
     * many small cells; sparse regions produce large cells.
     *
     * Format (quadtree.csv):  center_x,center_y,half_w,half_h
     *
     * @param tree      Fully built QuadTree on converged positions.
     * @param outputDir Directory to write into.
     * @param minMass   Minimum node count per cell (default 2).
     */
    static void exportQuadTree(const QuadTree& tree,
                               const fs::path& outputDir,
                               float minMass = 2.0f)
    {
        const std::vector<BoundingBox> boxes = tree.collectBoxes(minMass);

        const fs::path path = ensureDir(outputDir) / "quadtree.csv";
        std::ofstream  file = openFile(path);

        file << "center_x,center_y,half_w,half_h\n"
             << std::fixed << std::setprecision(4);

        for (const BoundingBox& b : boxes)
            file << b.center.x << ',' << b.center.y << ','
                 << b.halfW    << ',' << b.halfH     << '\n';

        checkStream(file, path);
        std::cout << "  QuadTree: " << boxes.size()
                  << " cells -> " << path << '\n';
    }

    // ── Animation frame writer (RAII) ─────────────────────────

    class AnimationWriter {
    public:
        explicit AnimationWriter(const fs::path& outputDir,
                                 bool overwrite = true)
        {
            ensureDir(outputDir);
            const fs::path path = outputDir / "animation_frames.csv";
            if (overwrite) {
                file_ = openFile(path);
            } else {
                file_.open(path, std::ios::app);
                if (!file_.is_open())
                    throw std::runtime_error(
                        "AnimationWriter: cannot open '" + path.string() + "'.");
                return;
            }
            file_ << "method,iteration,node_id,x,y\n"
                  << std::fixed << std::setprecision(6);
        }

        void appendFrame(const Graph& g, std::string_view method, int iter) {
            for (const Node& v : g.nodes())
                file_ << method << ',' << iter << ','
                      << v.id   << ',' << v.position.x << ','
                      << v.position.y  << '\n';
        }

        void flush() { file_.flush(); }
        ~AnimationWriter() { if (file_.is_open()) file_.flush(); }

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
        if (ec) throw std::runtime_error(
            "DataExporter: cannot create '" + dir.string() + "': " + ec.message());
        return dir;
    }
    static std::ofstream openFile(const fs::path& path) {
        std::ofstream f{ path };
        if (!f.is_open()) throw std::runtime_error(
            "DataExporter: cannot open '" + path.string() + "'.");
        return f;
    }
    static void checkStream(const std::ofstream& f, const fs::path& path) {
        if (!f.good()) throw std::runtime_error(
            "DataExporter: I/O error writing '" + path.string() + "'.");
    }
};
