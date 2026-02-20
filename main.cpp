#include "graph.hpp"
#include "layout_engine.hpp"

int main() {
    // Generate a random G(50, 0.15) graph
    Graph g = Graph::erdosRenyi(50, 0.15);

    // Create engine with a 800×600 frame
    LayoutEngine engine{ 800.0f, 600.0f, /*C=*/1.0f };
    engine.setTemperature(80.0f);
    engine.setCoolingRate(0.95f);
    engine.initialize(g);

    // Optionally swap to Barnes-Hut later:
    // engine.setRepulsiveStrategy(std::make_unique<BarnesHutRepulsion>(...));

    std::vector<float> convergenceCurve;
    for (int iter = 0; iter < 500; ++iter) {
        engine.step(g);
        convergenceCurve.push_back(engine.kineticEnergy());
    }
    // convergenceCurve is ready to plot
}