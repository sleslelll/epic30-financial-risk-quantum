from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EDGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "financial_network"
    / "latest_network_edges.csv"
)

COMMUNITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "community_detection"
    / "latest_communities.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "financial_network"
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load financial network edges and
    community assignments.
    """
    edges = pd.read_csv(EDGE_PATH)
    communities = pd.read_csv(COMMUNITY_PATH)

    return edges, communities


def build_graph(
    edges: pd.DataFrame,
) -> nx.Graph:
    """
    Build the full signed financial network.

    All pairwise asset relationships are retained.
    """
    graph = nx.Graph()

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            correlation=row["correlation"],
            distance=row["distance"],
        )

    return graph


def create_community_map(
    communities: pd.DataFrame,
) -> dict[str, int]:
    """
    Map each asset to its detected community.
    """
    return dict(
        zip(
            communities["asset"],
            communities["community"],
        )
    )


def calculate_layout(
    graph: nx.Graph,
) -> dict:
    """
    Calculate node positions using
    correlation-based distance.

    Shorter distance corresponds to
    stronger positive correlation.
    """
    distance_graph = nx.Graph()

    for u, v, data in graph.edges(data=True):
        distance_graph.add_edge(
            u,
            v,
            weight=data["distance"],
        )

    positions = nx.kamada_kawai_layout(
        distance_graph,
        weight="weight",
    )

    return positions


def plot_network(
    graph: nx.Graph,
    community_map: dict[str, int],
    output_path: Path,
) -> None:
    """
    Plot the latest financial network.

    Node position:
        Correlation-based distance.

    Edge width:
        Absolute correlation strength.

    Solid edges:
        Positive correlation.

    Dashed edges:
        Negative correlation.

    Node color:
        Detected community.
    """
    positions = calculate_layout(
        graph
    )

    positive_edges = [
        (u, v)
        for u, v, data in graph.edges(
            data=True
        )
        if data["correlation"] > 0
    ]

    negative_edges = [
        (u, v)
        for u, v, data in graph.edges(
            data=True
        )
        if data["correlation"] < 0
    ]

    positive_widths = [
        1
        + 4
        * abs(
            graph[u][v]["correlation"]
        )
        for u, v in positive_edges
    ]

    negative_widths = [
        1
        + 4
        * abs(
            graph[u][v]["correlation"]
        )
        for u, v in negative_edges
    ]

    node_communities = [
        community_map[node]
        for node in graph.nodes
    ]

    plt.figure(
        figsize=(10, 8)
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=1800,
        node_color=node_communities,
        cmap=plt.cm.Set2,
        edgecolors="black",
        linewidths=1.2,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=10,
        font_weight="bold",
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive_edges,
        width=positive_widths,
        alpha=0.65,
    )

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative_edges,
        width=negative_widths,
        alpha=0.55,
        style="dashed",
    )

    edge_labels = {
        (u, v): (
            f"{data['correlation']:.2f}"
        )
        for u, v, data in graph.edges(
            data=True
        )
        if abs(
            data["correlation"]
        ) >= 0.45
    }

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
    )

    plt.title(
        "EPIC30 Financial Network\n"
        "Latest 252-Day Correlation Structure"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edges, communities = load_data()

    graph = build_graph(
        edges
    )

    community_map = create_community_map(
        communities
    )

    output_path = (
        OUTPUT_DIR
        / "latest_financial_network.png"
    )

    plot_network(
        graph,
        community_map,
        output_path,
    )

    print(
        "EPIC30 Financial Network Visualization"
    )
    print("-" * 40)

    print(
        f"\nNodes: {graph.number_of_nodes()}"
    )

    print(
        f"Edges: {graph.number_of_edges()}"
    )

    print(
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    main()