from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


INPUT_PATH = Path(
    PROJECT_ROOT /
    "data" /
    "rolling_dependence" /
    "latest_correlation_window_252.csv"
)

OUTPUT_DIR = Path(
    PROJECT_ROOT /
    "data" /
    "financial_network"
)


def load_correlation_matrix(path: Path) -> pd.DataFrame:
    """
    Load the latest asset correlation matrix.
    """
    correlation = pd.read_csv(path, index_col=0)

    return correlation


def correlation_to_distance(
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert correlation into correlation distance.

    d_ij = sqrt(2 * (1 - rho_ij))
    """
    distance = np.sqrt(
        2 * (1 - correlation)
    )

    return pd.DataFrame(
        distance,
        index=correlation.index,
        columns=correlation.columns,
    )


def build_financial_network(
    correlation: pd.DataFrame,
    distance: pd.DataFrame,
) -> nx.Graph:
    """
    Build an undirected weighted financial network.

    Nodes:
        Financial assets.

    Edges:
        Pairwise asset relationships.

    Attributes:
        correlation: signed correlation coefficient
        absolute_correlation: correlation strength
        distance: correlation-based distance
    """
    graph = nx.Graph()

    assets = correlation.columns.tolist()

    graph.add_nodes_from(assets)

    for i, asset_i in enumerate(assets):
        for j in range(i + 1, len(assets)):
            asset_j = assets[j]

            rho = correlation.loc[
                asset_i,
                asset_j,
            ]

            dist = distance.loc[
                asset_i,
                asset_j,
            ]

            graph.add_edge(
                asset_i,
                asset_j,
                correlation=rho,
                absolute_correlation=abs(rho),
                distance=dist,
            )

    return graph


def create_edge_table(
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Convert graph edges into a tabular summary.
    """
    rows = []

    for source, target, data in graph.edges(data=True):
        rows.append(
            {
                "source": source,
                "target": target,
                "correlation": data["correlation"],
                "absolute_correlation": data[
                    "absolute_correlation"
                ],
                "distance": data["distance"],
            }
        )

    edge_table = pd.DataFrame(rows)

    return edge_table.sort_values(
        "absolute_correlation",
        ascending=False,
    )


def calculate_node_strength(
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Calculate weighted node strength using
    absolute correlations.
    """
    rows = []

    for node in graph.nodes:
        strength = sum(
            data["absolute_correlation"]
            for _, _, data in graph.edges(
                node,
                data=True,
            )
        )

        rows.append(
            {
                "asset": node,
                "strength": strength,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "strength",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    correlation = load_correlation_matrix(
        INPUT_PATH
    )

    distance = correlation_to_distance(
        correlation
    )

    graph = build_financial_network(
        correlation,
        distance,
    )

    edge_table = create_edge_table(
        graph
    )

    node_strength = calculate_node_strength(
        graph
    )

    distance.to_csv(
        OUTPUT_DIR
        / "latest_distance_matrix.csv"
    )

    edge_table.to_csv(
        OUTPUT_DIR
        / "latest_network_edges.csv",
        index=False,
    )

    node_strength.to_csv(
        OUTPUT_DIR
        / "latest_node_strength.csv",
        index=False,
    )

    print(
        "EPIC30 Financial Network Analysis"
    )
    print("-" * 40)

    print(
        f"\nNodes: {graph.number_of_nodes()}"
    )

    print(
        f"Edges: {graph.number_of_edges()}"
    )

    print(
        "\nStrongest Relationships"
    )
    print("-" * 40)

    print(
        edge_table.head(10).to_string(
            index=False
        )
    )

    print(
        "\nNode Strength"
    )
    print("-" * 40)

    print(
        node_strength.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()