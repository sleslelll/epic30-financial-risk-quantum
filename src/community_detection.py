from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


INPUT_PATH = Path(
    PROJECT_ROOT /
    "data" /
    "financial_network" /
    "latest_network_edges.csv"
)


OUTPUT_DIR = Path(
    PROJECT_ROOT /
    "data" /
    "community_detection"
)


def load_edge_table(path: Path) -> pd.DataFrame:
    """
    Load the latest financial network edge table.
    """
    return pd.read_csv(path)


def build_positive_network(
    edge_table: pd.DataFrame,
) -> nx.Graph:
    """
    Build a positive-correlation similarity network.

    Only positive correlations are retained because
    standard community detection methods assume
    non-negative similarity weights.
    """
    graph = nx.Graph()

    assets = sorted(
        set(edge_table["source"])
        | set(edge_table["target"])
    )

    graph.add_nodes_from(assets)

    positive_edges = edge_table[
        edge_table["correlation"] > 0
    ]

    for _, row in positive_edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            weight=row["correlation"],
            correlation=row["correlation"],
            distance=row["distance"],
        )

    return graph


def detect_communities(
    graph: nx.Graph,
) -> list[set[str]]:
    """
    Detect communities using greedy modularity
    maximization with correlation as edge weight.
    """
    communities = (
        nx.community.greedy_modularity_communities(
            graph,
            weight="weight",
        )
    )

    return [
        set(community)
        for community in communities
    ]


def create_community_table(
    communities: list[set[str]],
) -> pd.DataFrame:
    """
    Convert detected communities into a table.
    """
    rows = []

    for community_id, community in enumerate(
        communities,
        start=1,
    ):
        for asset in sorted(community):
            rows.append(
                {
                    "asset": asset,
                    "community": community_id,
                }
            )

    return pd.DataFrame(rows)


def create_community_edge_table(
    graph: nx.Graph,
    community_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add community labels to network edges.
    """
    community_map = dict(
        zip(
            community_table["asset"],
            community_table["community"],
        )
    )

    rows = []

    for source, target, data in graph.edges(
        data=True
    ):
        source_community = community_map[source]
        target_community = community_map[target]

        rows.append(
            {
                "source": source,
                "target": target,
                "correlation": data[
                    "correlation"
                ],
                "distance": data[
                    "distance"
                ],
                "source_community":
                    source_community,
                "target_community":
                    target_community,
                "same_community":
                    source_community
                    == target_community,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_modularity(
    graph: nx.Graph,
    communities: list[set[str]],
) -> float:
    """
    Calculate weighted modularity of the
    detected community partition.
    """
    return nx.community.modularity(
        graph,
        communities,
        weight="weight",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_table = load_edge_table(
        INPUT_PATH
    )

    graph = build_positive_network(
        edge_table
    )

    communities = detect_communities(
        graph
    )

    community_table = create_community_table(
        communities
    )

    community_edges = create_community_edge_table(
        graph,
        community_table,
    )

    modularity = calculate_modularity(
        graph,
        communities,
    )

    community_table.to_csv(
        OUTPUT_DIR
        / "latest_communities.csv",
        index=False,
    )

    community_edges.to_csv(
        OUTPUT_DIR
        / "latest_community_edges.csv",
        index=False,
    )

    print(
        "EPIC30 Community Detection"
    )
    print("-" * 40)

    print(
        f"\nNodes: {graph.number_of_nodes()}"
    )

    print(
        f"Positive edges: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Communities: {len(communities)}"
    )

    print(
        f"Modularity: {modularity:.4f}"
    )

    print(
        "\nDetected Communities"
    )
    print("-" * 40)

    for community_id, community in enumerate(
        communities,
        start=1,
    ):
        assets = ", ".join(
            sorted(community)
        )

        print(
            f"Community {community_id}: "
            f"{assets}"
        )


if __name__ == "__main__":
    main()