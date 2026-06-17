/**
 * batch.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Batch layout processor.
 *
 * Reads every .txt adjacency-list file from the Input/ directory,
 * runs the Fruchterman-Reingold layout algorithm on each graph,
 * and writes per-graph results to Output/<graph_name>/.
 *
 * Usage:
 *   ./build/fr_batch                    # uses ./Input and ./Output
 *   ./build/fr_batch MyInput MyOutput   # custom directories
 *
 * Output per graph (in Output/<stem>/):
 *   nodes.csv   — final node positions (node_id, x, y)
 *   edges.csv   — edge list            (source, target)
 *
 * Then run:
 *   python batch_visualise.py
 * to produce a layout PDF for every graph in Output/.
 */

#include "graph.hpp"
#include "layout_engine.hpp"
#include "barnes_hut.hpp"
#include "exporter.hpp"
#include "parser.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <string>

namespace fs  = std::filesystem;
namespace chr = std::chrono;

// ── Layout configuration ──────────────────────────────────────────────────────

struct LayoutConfig {
    float frameW      = 1920.0f;
    float frameH      = 1920.0f;
    float C           = 0.8f;     // DECREASED: Usually 0.5 to 1.2 is the sweet spot. Try 0.8.
    float initTemp    = 192.0f;   // DECREASED: A good rule of thumb is frameW / 10.0f
    float coolingRate = 0.99f;    // KEEP: Slow cooling is perfect for symmetry.
    float theta       = 0.5f;     // KEEP: Standard for Barnes-Hut.
    int   iterations  = 2000;     // KEEP: Let it run long enough to untangle.
    std::uint64_t layoutSeed = 42;

    // Optional I/O overrides (only set when supplied via a config file).
    fs::path inputDir  = "Input";
    fs::path outputDir = "Output";
};

// ── Config file loader ────────────────────────────────────────────────────────

/**
 * Loads a `key=value` config file (as written by the UI control panel) and
 * overrides the matching LayoutConfig fields. Unknown keys are ignored, blank
 * lines and lines starting with '#' are skipped. Any field not present in the
 * file keeps its hardcoded default, so the file may be partial.
 *
 * Recognised keys:
 *   frameW frameH C initTemp coolingRate theta iterations (alias: maxIter)
 *   layoutSeed inputDir outputDir
 */
static LayoutConfig loadConfig(const fs::path& path) {
    LayoutConfig cfg;

    std::ifstream file{ path };
    if (!file.is_open())
        throw std::runtime_error(
            "loadConfig: cannot open config file '" + path.string() + "'.");

    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        ++lineNum;
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Strip a leading UTF-8 BOM (EF BB BF) on the first line, which some
        // editors / PowerShell add and which would otherwise break parsing.
        if (lineNum == 1 && line.size() >= 3 &&
            static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF)
            line.erase(0, 3);

        // Strip leading whitespace, skip blanks and comments.
        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        if (line[first] == '#') continue;

        const auto eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error(
                "loadConfig: missing '=' on line " + std::to_string(lineNum) +
                " of '" + path.string() + "'.");

        auto trim = [](std::string s) {
            const auto b = s.find_first_not_of(" \t");
            if (b == std::string::npos) return std::string{};
            const auto e = s.find_last_not_of(" \t");
            return s.substr(b, e - b + 1);
        };

        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        if (key.empty() || val.empty()) continue;

        try {
            if      (key == "frameW")      cfg.frameW      = std::stof(val);
            else if (key == "frameH")      cfg.frameH      = std::stof(val);
            else if (key == "C")           cfg.C           = std::stof(val);
            else if (key == "initTemp")    cfg.initTemp    = std::stof(val);
            else if (key == "coolingRate") cfg.coolingRate = std::stof(val);
            else if (key == "theta")       cfg.theta       = std::stof(val);
            else if (key == "iterations" ||
                     key == "maxIter")     cfg.iterations  = std::stoi(val);
            else if (key == "layoutSeed")  cfg.layoutSeed  =
                         static_cast<std::uint64_t>(std::stoull(val));
            else if (key == "inputDir")    cfg.inputDir    = val;
            else if (key == "outputDir")   cfg.outputDir   = val;
            // Unknown keys (e.g. animation-only frameInterval/graphSeed) ignored.
        } catch (const std::exception&) {
            throw std::runtime_error(
                "loadConfig: invalid value '" + val + "' for key '" + key +
                "' on line " + std::to_string(lineNum) + ".");
        }
    }

    return cfg;
}

// ── Layout one graph ──────────────────────────────────────────────────────────

/**
 * Runs the FR layout on a single graph and exports nodes.csv + edges.csv
 * to the specified output directory.
 *
 * Uses Barnes-Hut at theta=0.5 for a good accuracy/speed balance.
 * For small graphs (|V| < 200) this is slightly slower than brute force
 * but produces identical-quality layouts.
 */
static void layoutAndExport(Graph&             g,
                             const fs::path&    outputDir,
                             const LayoutConfig& cfg)
{
    // Choose strategy based on graph size:
    // Barnes-Hut is beneficial above ~200 nodes.
    LayoutEngine engine{ cfg.frameW, cfg.frameH, cfg.C };
    engine.setTemperature(cfg.initTemp);
    engine.setCoolingRate(cfg.coolingRate);

    if (g.vertexCount() > 200) {
        engine.setRepulsiveStrategy(
            std::make_unique<BarnesHutRepulsion>(cfg.theta));
    }
    // else: default BruteForceRepulsion

    engine.initialize(g, cfg.layoutSeed);

    const auto t0 = chr::high_resolution_clock::now();
    for (int i = 0; i < cfg.iterations; ++i)
        engine.step(g);
    const auto t1 = chr::high_resolution_clock::now();

    const auto ms = chr::duration_cast<chr::milliseconds>(t1 - t0).count();

    DataExporter::exportNodes(g, outputDir);
    DataExporter::exportEdges(g, outputDir);

    std::cout << "  Layout done in " << ms << " ms"
              << "  ->  " << outputDir.string() << '\n';
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Argument handling (backward compatible):
    //   fr_batch                       -> Input/  Output/  with default config
    //   fr_batch <inputDir> [outDir]   -> custom dirs, default config
    //   fr_batch <config.txt>          -> all settings + dirs from config file
    //                                     (the UI control panel uses this form)
    LayoutConfig cfg;
    fs::path inputDir  = cfg.inputDir;
    fs::path outputDir = cfg.outputDir;

    if (argc > 1 && fs::is_regular_file(argv[1])) {
        try {
            cfg = loadConfig(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] " << e.what() << '\n';
            return EXIT_FAILURE;
        }
        inputDir  = cfg.inputDir;
        outputDir = cfg.outputDir;
    } else {
        if (argc > 1) inputDir  = argv[1];
        if (argc > 2) outputDir = argv[2];
    }

    std::cout << "FR Batch Layout Processor\n"
              << "=========================\n"
              << "Input  : " << fs::absolute(inputDir)  << '\n'
              << "Output : " << fs::absolute(outputDir) << '\n'
              << "Iters  : " << cfg.iterations
              << "   theta : " << cfg.theta << "\n\n";

    // ── Parse all .txt files from Input/ ─────────────────────
    std::cout << "Parsing input files ...\n";

    std::vector<std::pair<std::string, Graph>> graphs;
    try {
        graphs = AdjacencyListParser::parseDirectory(inputDir);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    if (graphs.empty()) {
        std::cerr << "[ERROR] No .txt files found in '"
                  << inputDir.string() << "'.\n"
                  << "  Place adjacency-list .txt files in the Input/ folder.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nFound " << graphs.size() << " graph(s). Running layout ...\n\n";

    // ── Layout each graph ─────────────────────────────────────
    int success = 0;
    int failure = 0;

    for (auto& [name, g] : graphs) {
        std::cout << "[ " << name << " ]\n"
                  << "  |V| = " << g.vertexCount()
                  << "   |E| = " << g.edgeCount() << '\n';

        const fs::path graphOutputDir = outputDir / name;
        fs::create_directories(graphOutputDir);

        try {
            layoutAndExport(g, graphOutputDir, cfg);
            ++success;
        } catch (const std::exception& e) {
            std::cerr << "  [ERROR] " << e.what() << '\n';
            ++failure;
        }

        std::cout << '\n';
    }

    // ── Summary ───────────────────────────────────────────────
    std::cout << "Done.  " << success << " succeeded";
    if (failure > 0) std::cout << "  " << failure << " failed";
    std::cout << "\n\nNext step:\n"
              << "  python batch_visualise.py\n";

    return (failure == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}