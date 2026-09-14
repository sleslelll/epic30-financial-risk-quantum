from pathlib import Path

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
    / "systemic_risk"
)


ATTENUATION = 0.5
PROPAGATION_STEPS = 5
INITIAL_SHOCK = 1.0


def load_edge_table(
    path: Path,
) -> pd.DataFrame:
    """
    Load the latest financial network edge table.
    """
    return pd.read_csv(path)


def get_assets(
    edge_table: pd.DataFrame,
) -> list[str]:
    """
    Extract the complete sorted asset list.
    """
    return sorted(
        set(edge_table["source"])
        | set(edge_table["target"])
    )


def create_dependence_matrix(
    edge_table: pd.DataFrame,
    assets: list[str],
) -> pd.DataFrame:
    """
    Create a symmetric dependence matrix using
    absolute correlation as edge strength.
    """
    matrix = pd.DataFrame(
        0.0,
        index=assets,
        columns=assets,
    )

    for _, row in edge_table.iterrows():
        source = row["source"]
        target = row["target"]

        weight = abs(
            row["correlation"]
        )

        matrix.loc[
            source,
            target,
        ] = weight

        matrix.loc[
            target,
            source,
        ] = weight

    return matrix


def calculate_spectral_radius(
    dependence: pd.DataFrame,
) -> float:
    """
    Calculate the spectral radius of the
    symmetric dependence matrix.
    """
    eigenvalues = np.linalg.eigvalsh(
        dependence.to_numpy()
    )

    spectral_radius = float(
        np.max(
            np.abs(
                eigenvalues
            )
        )
    )

    if spectral_radius <= 0:
        raise ValueError(
            "Spectral radius must be positive."
        )

    return spectral_radius


def create_transmission_matrix(
    dependence: pd.DataFrame,
    spectral_radius: float,
) -> pd.DataFrame:
    """
    Scale the entire dependence matrix by one
    common spectral-radius factor.

    Global scaling preserves relative differences
    in node connectivity.
    """
    return (
        dependence
        / spectral_radius
    )


def propagate_single_shock(
    source: str,
    assets: list[str],
    transmission: pd.DataFrame,
) -> tuple[np.ndarray, list[dict]]:
    """
    Propagate one unit shock from a source asset
    through the network for a fixed number of steps.
    """
    source_index = assets.index(
        source
    )

    current_shock = np.zeros(
        len(assets),
        dtype=float,
    )

    current_shock[
        source_index
    ] = INITIAL_SHOCK

    cumulative_propagated = np.zeros(
        len(assets),
        dtype=float,
    )

    path_rows = []

    transmission_array = (
        transmission.to_numpy()
    )

    for step in range(
        1,
        PROPAGATION_STEPS + 1,
    ):
        next_shock = (
            ATTENUATION
            * transmission_array.T
            @ current_shock
        )

        cumulative_propagated += (
            next_shock
        )

        for asset, shock in zip(
            assets,
            next_shock,
        ):
            path_rows.append(
                {
                    "shock_source": source,
                    "step": step,
                    "asset": asset,
                    "shock": shock,
                }
            )

        current_shock = next_shock

    return (
        cumulative_propagated,
        path_rows,
    )


def calculate_systemic_risk(
    assets: list[str],
    transmission: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Shock each asset separately and calculate
    cumulative network propagation.
    """
    propagation_rows = []
    score_rows = []
    path_rows = []

    for source in assets:
        (
            cumulative,
            source_paths,
        ) = propagate_single_shock(
            source,
            assets,
            transmission,
        )

        source_index = assets.index(
            source
        )

        total_propagated_shock = float(
            cumulative.sum()
        )

        source_reverberation = float(
            cumulative[source_index]
        )

        external_system_impact = float(
            total_propagated_shock
            - source_reverberation
        )

        score_rows.append(
            {
                "shock_source": source,
                "total_propagated_shock":
                    total_propagated_shock,
                "external_system_impact":
                    external_system_impact,
                "source_reverberation":
                    source_reverberation,
            }
        )

        propagation_row = {
            "shock_source": source
        }

        for asset, impact in zip(
            assets,
            cumulative,
        ):
            propagation_row[
                asset
            ] = impact

        propagation_rows.append(
            propagation_row
        )

        path_rows.extend(
            source_paths
        )

    scores = pd.DataFrame(
        score_rows
    ).sort_values(
        "total_propagated_shock",
        ascending=False,
    ).reset_index(
        drop=True
    )

    scores[
        "systemic_risk_rank"
    ] = (
        scores.index + 1
    )

    propagation_matrix = pd.DataFrame(
        propagation_rows
    )

    shock_paths = pd.DataFrame(
        path_rows
    )

    return (
        scores,
        propagation_matrix,
        shock_paths,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edge_table = load_edge_table(
        EDGE_PATH
    )

    assets = get_assets(
        edge_table
    )

    dependence = (
        create_dependence_matrix(
            edge_table,
            assets,
        )
    )

    spectral_radius = (
        calculate_spectral_radius(
            dependence
        )
    )

    transmission = (
        create_transmission_matrix(
            dependence,
            spectral_radius,
        )
    )

    (
        systemic_scores,
        propagation_matrix,
        shock_paths,
    ) = calculate_systemic_risk(
        assets,
        transmission,
    )

    dependence.to_csv(
        OUTPUT_DIR
        / "dependence_matrix.csv"
    )

    transmission.to_csv(
        OUTPUT_DIR
        / "transmission_matrix.csv"
    )

    systemic_scores.to_csv(
        OUTPUT_DIR
        / "systemic_risk_scores.csv",
        index=False,
    )

    propagation_matrix.to_csv(
        OUTPUT_DIR
        / "shock_propagation_matrix.csv",
        index=False,
    )

    shock_paths.to_csv(
        OUTPUT_DIR
        / "shock_paths.csv",
        index=False,
    )

    print(
        "EPIC30 Systemic Risk Analysis"
    )
    print("-" * 40)

    print(
        f"\nAssets: {len(assets)}"
    )

    print(
        f"Propagation steps: "
        f"{PROPAGATION_STEPS}"
    )

    print(
        f"Attenuation: "
        f"{ATTENUATION:.2f}"
    )

    print(
        f"Initial shock: "
        f"{INITIAL_SHOCK:.2f}"
    )

    print(
        f"Spectral radius: "
        f"{spectral_radius:.6f}"
    )

    print(
        "\nSystemic Risk Ranking"
    )
    print("-" * 40)

    print(
        systemic_scores.to_string(
            index=False
        )
    )

    print(
        "\nDependence Matrix"
    )
    print("-" * 40)

    print(
        dependence.round(
            4
        ).to_string()
    )

    print(
        "\nTransmission Matrix"
    )
    print("-" * 40)

    print(
        transmission.round(
            4
        ).to_string()
    )


if __name__ == "__main__":
    main()