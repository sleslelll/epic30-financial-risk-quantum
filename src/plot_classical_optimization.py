from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "classical_optimization"
    / "systemic_risk_sensitivity.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "classical_optimization"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "systemic_risk_weight_sensitivity.png"
)


ASSETS = [
    "ETH",
    "BTC",
    "NASDAQ",
    "SP500",
    "GOLD",
    "USD",
]

ACTIVE_WEIGHT_THRESHOLD = 1e-4


def load_sensitivity_results(
    path: Path,
) -> pd.DataFrame:
    """
    Load systemic-risk sensitivity results.
    """
    results = pd.read_csv(path)

    required_columns = [
        "systemic_risk_weight",
        *ASSETS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return results.sort_values(
        "systemic_risk_weight"
    ).reset_index(
        drop=True
    )


def get_active_assets(
    row: pd.Series,
) -> list[str]:
    """
    Identify assets with economically non-zero weights.
    """
    return [
        asset
        for asset in ASSETS
        if row[asset] > ACTIVE_WEIGHT_THRESHOLD
    ]


def find_regime_transitions(
    results: pd.DataFrame,
) -> list[dict]:
    """
    Identify points where the set of active assets changes.
    """
    transitions = []

    previous_assets = None

    for _, row in results.iterrows():
        active_assets = get_active_assets(
            row
        )

        current_assets = tuple(
            active_assets
        )

        if current_assets != previous_assets:
            transitions.append(
                {
                    "eta":
                        row[
                            "systemic_risk_weight"
                        ],
                    "assets":
                        active_assets,
                }
            )

            previous_assets = current_assets

    return transitions


def plot_sensitivity(
    results: pd.DataFrame,
    transitions: list[dict],
    output_path: Path,
) -> None:
    """
    Plot portfolio weights across systemic-risk penalties.
    """
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    eta = results[
        "systemic_risk_weight"
    ]

    for asset in ASSETS:
        ax.plot(
            eta,
            results[asset],
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=asset,
        )

    for transition in transitions:
        transition_eta = transition[
            "eta"
        ]

        active_assets = transition[
            "assets"
        ]

        ax.axvline(
            x=transition_eta,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )

        label = (
            " + ".join(active_assets)
        )

        ax.text(
            transition_eta,
            1.02,
            label,
            rotation=45,
            ha="left",
            va="bottom",
            fontsize=8,
            transform=ax.get_xaxis_transform(),
        )

    ax.set_title(
        "Portfolio Allocation Across "
        "Systemic-Risk Penalties"
    )

    ax.set_xlabel(
        "Systemic-Risk Penalty (η)"
    )

    ax.set_ylabel(
        "Portfolio Weight"
    )

    ax.set_xlim(
        results[
            "systemic_risk_weight"
        ].min(),
        results[
            "systemic_risk_weight"
        ].max(),
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title="Asset",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_sensitivity_results(
        INPUT_PATH
    )

    transitions = find_regime_transitions(
        results
    )

    plot_sensitivity(
        results,
        transitions,
        OUTPUT_PATH,
    )

    print(
        "EPIC30 Classical Optimization Visualization"
    )
    print("-" * 60)

    print(
        f"\nSensitivity observations: "
        f"{len(results)}"
    )

    print(
        f"Regime transitions: "
        f"{len(transitions)}"
    )

    print(
        "\nPortfolio Regimes"
    )
    print("-" * 60)

    for transition in transitions:
        eta = transition[
            "eta"
        ]

        assets = " + ".join(
            transition[
                "assets"
            ]
        )

        print(
            f"eta = {eta:.3f}: "
            f"{assets}"
        )

    print(
        f"\nSaved figure: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()