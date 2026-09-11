from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


EDGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "financial_network"
    / "latest_network_edges.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "spectral_analysis"
)

TOLERANCE = 1e-10


def load_edge_table(
    path: Path,
) -> pd.DataFrame:
    """
    Load the latest financial network edge table.
    """
    return pd.read_csv(path)


def build_positive_graph(
    edge_table: pd.DataFrame,
) -> nx.Graph:
    """
    Build a weighted graph using only positive correlations.

    Standard graph Laplacian analysis assumes
    non-negative edge weights.
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
        )

    return graph


def create_adjacency_matrix(
    graph: nx.Graph,
    nodes: list[str],
) -> pd.DataFrame:
    """
    Create the weighted adjacency matrix.
    """
    matrix = nx.to_numpy_array(
        graph,
        nodelist=nodes,
        weight="weight",
        dtype=float,
    )

    return pd.DataFrame(
        matrix,
        index=nodes,
        columns=nodes,
    )


def create_degree_matrix(
    adjacency: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the weighted degree matrix.
    """
    degrees = adjacency.sum(
        axis=1
    ).to_numpy()

    matrix = np.diag(
        degrees
    )

    return pd.DataFrame(
        matrix,
        index=adjacency.index,
        columns=adjacency.columns,
    )


def create_laplacian_matrix(
    adjacency: pd.DataFrame,
    degree: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the unnormalized graph Laplacian.

    L = D - A
    """
    matrix = (
        degree.to_numpy()
        - adjacency.to_numpy()
    )

    return pd.DataFrame(
        matrix,
        index=adjacency.index,
        columns=adjacency.columns,
    )


def calculate_eigendecomposition(
    laplacian: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate sorted eigenvalues and eigenvectors
    of a symmetric Laplacian matrix.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        laplacian.to_numpy()
    )

    order = np.argsort(
        eigenvalues
    )

    return (
        eigenvalues[order],
        eigenvectors[:, order],
    )


def create_eigenvalue_table(
    eigenvalues: np.ndarray,
) -> pd.DataFrame:
    """
    Create a table of Laplacian eigenvalues.
    """
    return pd.DataFrame(
        {
            "eigen_index": range(
                len(eigenvalues)
            ),
            "eigenvalue": eigenvalues,
            "is_zero": np.abs(
                eigenvalues
            ) < TOLERANCE,
        }
    )


def create_eigenvector_table(
    nodes: list[str],
    eigenvectors: np.ndarray,
) -> pd.DataFrame:
    """
    Create a table containing all eigenvectors.
    """
    data = {
        "asset": nodes
    }

    for index in range(
        eigenvectors.shape[1]
    ):
        data[
            f"eigenvector_{index}"
        ] = eigenvectors[
            :,
            index
        ]

    return pd.DataFrame(
        data
    )


def create_component_table(
    graph: nx.Graph,
) -> pd.DataFrame:
    """
    Identify connected components in the
    positive-correlation network.
    """
    components = sorted(
        nx.connected_components(graph),
        key=len,
        reverse=True,
    )

    rows = []

    for component_id, component in enumerate(
        components,
        start=1,
    ):
        for asset in sorted(component):
            rows.append(
                {
                    "asset": asset,
                    "component": component_id,
                    "component_size": len(component),
                }
            )

    return pd.DataFrame(
        rows
    )


def analyze_component(
    graph: nx.Graph,
    component_id: int,
    nodes: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Perform spectral analysis within one
    connected component.

    For components with at least two nodes,
    extract the component-level Fiedler vector.
    """
    subgraph = graph.subgraph(
        nodes
    ).copy()

    adjacency = create_adjacency_matrix(
        subgraph,
        nodes,
    )

    degree = create_degree_matrix(
        adjacency
    )

    laplacian = create_laplacian_matrix(
        adjacency,
        degree,
    )

    eigenvalues, eigenvectors = (
        calculate_eigendecomposition(
            laplacian
        )
    )

    rows = []

    if len(nodes) >= 2:
        fiedler_value = eigenvalues[1]
        fiedler_vector = eigenvectors[:, 1]

        for asset, vector_value in zip(
            nodes,
            fiedler_vector,
        ):
            if vector_value > TOLERANCE:
                spectral_group = 1
            elif vector_value < -TOLERANCE:
                spectral_group = 2
            else:
                spectral_group = 0

            rows.append(
                {
                    "component": component_id,
                    "asset": asset,
                    "fiedler_value": vector_value,
                    "spectral_group": spectral_group,
                }
            )

        algebraic_connectivity = float(
            fiedler_value
        )

    else:
        rows.append(
            {
                "component": component_id,
                "asset": nodes[0],
                "fiedler_value": np.nan,
                "spectral_group": 0,
            }
        )

        algebraic_connectivity = np.nan

    summary = {
        "component": component_id,
        "nodes": len(nodes),
        "edges": subgraph.number_of_edges(),
        "algebraic_connectivity":
            algebraic_connectivity,
    }

    return (
        pd.DataFrame(rows),
        summary,
    )


def analyze_components(
    graph: nx.Graph,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform separate Fiedler analysis for
    every connected component.
    """
    components = sorted(
        nx.connected_components(graph),
        key=len,
        reverse=True,
    )

    fiedler_tables = []
    summaries = []

    for component_id, component in enumerate(
        components,
        start=1,
    ):
        nodes = sorted(
            component
        )

        fiedler_table, summary = (
            analyze_component(
                graph,
                component_id,
                nodes,
            )
        )

        fiedler_tables.append(
            fiedler_table
        )

        summaries.append(
            summary
        )

    return (
        pd.concat(
            fiedler_tables,
            ignore_index=True,
        ),
        pd.DataFrame(
            summaries
        ),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_table = load_edge_table(
        EDGE_PATH
    )

    graph = build_positive_graph(
        edge_table
    )

    nodes = sorted(
        graph.nodes()
    )

    adjacency = create_adjacency_matrix(
        graph,
        nodes,
    )

    degree = create_degree_matrix(
        adjacency
    )

    laplacian = create_laplacian_matrix(
        adjacency,
        degree,
    )

    eigenvalues, eigenvectors = (
        calculate_eigendecomposition(
            laplacian
        )
    )

    eigenvalue_table = (
        create_eigenvalue_table(
            eigenvalues
        )
    )

    eigenvector_table = (
        create_eigenvector_table(
            nodes,
            eigenvectors,
        )
    )

    component_table = (
        create_component_table(
            graph
        )
    )

    (
        component_fiedler_table,
        component_summary,
    ) = analyze_components(
        graph
    )

    zero_eigenvalues = int(
        np.sum(
            np.abs(
                eigenvalues
            ) < TOLERANCE
        )
    )

    connected_components = (
        nx.number_connected_components(
            graph
        )
    )

    global_algebraic_connectivity = (
        float(eigenvalues[1])
        if len(eigenvalues) > 1
        else np.nan
    )

    adjacency.to_csv(
        OUTPUT_DIR
        / "latest_adjacency_matrix.csv"
    )

    degree.to_csv(
        OUTPUT_DIR
        / "latest_degree_matrix.csv"
    )

    laplacian.to_csv(
        OUTPUT_DIR
        / "latest_laplacian_matrix.csv"
    )

    eigenvalue_table.to_csv(
        OUTPUT_DIR
        / "latest_eigenvalues.csv",
        index=False,
    )

    eigenvector_table.to_csv(
        OUTPUT_DIR
        / "latest_eigenvectors.csv",
        index=False,
    )

    component_table.to_csv(
        OUTPUT_DIR
        / "latest_connected_components.csv",
        index=False,
    )

    component_fiedler_table.to_csv(
        OUTPUT_DIR
        / "latest_component_fiedler_vectors.csv",
        index=False,
    )

    component_summary.to_csv(
        OUTPUT_DIR
        / "latest_component_spectral_summary.csv",
        index=False,
    )

    print(
        "EPIC30 Spectral Network Analysis"
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
        f"Connected components: "
        f"{connected_components}"
    )

    print(
        f"Zero eigenvalues: "
        f"{zero_eigenvalues}"
    )

    print(
        "\nEigenvalues"
    )
    print("-" * 40)

    print(
        eigenvalue_table.to_string(
            index=False
        )
    )

    print(
        "\nConnected Components"
    )
    print("-" * 40)

    print(
        component_table.to_string(
            index=False
        )
    )

    print(
        "\nGlobal Algebraic Connectivity"
    )
    print("-" * 40)

    if (
        abs(
            global_algebraic_connectivity
        )
        < TOLERANCE
    ):
        print(
            "0.000000"
        )

        print(
            "The positive-correlation graph "
            "is disconnected."
        )

        print(
            "Global Fiedler bipartition "
            "is therefore not applied."
        )

    else:
        print(
            f"{global_algebraic_connectivity:.6f}"
        )

    print(
        "\nComponent Spectral Analysis"
    )
    print("-" * 40)

    print(
        component_summary.to_string(
            index=False
        )
    )

    print(
        "\nComponent Fiedler Vectors"
    )
    print("-" * 40)

    print(
        component_fiedler_table.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()