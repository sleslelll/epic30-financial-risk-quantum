from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETURNS_PATH = (
    PROJECT_ROOT
    / "data"
    / "market_returns.csv"
)

COVARIANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "rolling_dependence"
    / "latest_covariance_window_252.csv"
)

SYSTEMIC_RISK_PATH = (
    PROJECT_ROOT
    / "data"
    / "systemic_risk"
    / "systemic_risk_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "qubo"
)


INVESTABLE_ASSETS = [
    "ETH",
    "BTC",
    "NASDAQ",
    "SP500",
    "GOLD",
    "USD",
]

NUMBER_OF_SELECTED_ASSETS = 3

RISK_AVERSION = 0.5
RETURN_WEIGHT = 0.5

STANDARD_ETA = 0.0
SYSTEMIC_ETA = 0.15

TRADING_DAYS = 252

PENALTY_VALUES = [
    0.010,
    0.012,
    0.014,
    0.016,
    0.018,
    0.020,
    0.022,
    0.024,
    0.025,
]


def load_returns(
    path: Path,
) -> pd.DataFrame:
    """
    Load market returns and retain numeric columns.
    """
    returns = pd.read_csv(path)

    if "Date" in returns.columns:
        returns = returns.drop(
            columns=["Date"]
        )

    return returns.select_dtypes(
        include=[np.number]
    )


def load_covariance(
    path: Path,
) -> pd.DataFrame:
    """
    Load the latest rolling covariance matrix.
    """
    return pd.read_csv(
        path,
        index_col=0,
    )


def load_systemic_risk(
    path: Path,
) -> pd.Series:
    """
    Load systemic risk scores indexed by network node.
    """
    scores = pd.read_csv(path)

    return scores.set_index(
        "shock_source"
    )[
        "total_propagated_shock"
    ]


def prepare_inputs():
    """
    Prepare expected returns, covariance,
    and scaled systemic risk inputs.
    """
    returns = load_returns(
        RETURNS_PATH
    )

    covariance = load_covariance(
        COVARIANCE_PATH
    )

    systemic_risk = load_systemic_risk(
        SYSTEMIC_RISK_PATH
    )

    missing_returns = [
        asset
        for asset in INVESTABLE_ASSETS
        if asset not in returns.columns
    ]

    missing_covariance = [
        asset
        for asset in INVESTABLE_ASSETS
        if asset not in covariance.index
        or asset not in covariance.columns
    ]

    missing_systemic = [
        asset
        for asset in INVESTABLE_ASSETS
        if asset not in systemic_risk.index
    ]

    if missing_returns:
        raise ValueError(
            "Missing assets in returns: "
            f"{missing_returns}"
        )

    if missing_covariance:
        raise ValueError(
            "Missing assets in covariance: "
            f"{missing_covariance}"
        )

    if missing_systemic:
        raise ValueError(
            "Missing assets in systemic risk: "
            f"{missing_systemic}"
        )

    expected_returns = (
        returns[INVESTABLE_ASSETS].mean()
        * TRADING_DAYS
    )

    covariance = (
        covariance.loc[
            INVESTABLE_ASSETS,
            INVESTABLE_ASSETS,
        ]
        * TRADING_DAYS
    )

    systemic_risk = systemic_risk.reindex(
        INVESTABLE_ASSETS
    )

    systemic_risk = (
        systemic_risk
        / systemic_risk.max()
    )

    return (
        expected_returns,
        covariance,
        systemic_risk,
    )


def original_objective(
    x: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    eta: float,
) -> float:
    """
    Evaluate the original discrete portfolio objective.
    """
    weights = (
        x
        / NUMBER_OF_SELECTED_ASSETS
    )

    variance = float(
        weights
        @ covariance
        @ weights
    )

    expected_return = float(
        weights
        @ expected_returns
    )

    systemic_exposure = float(
        weights
        @ systemic_risk
    )

    return (
        RISK_AVERSION * variance
        - RETURN_WEIGHT * expected_return
        + eta * systemic_exposure
    )


def build_qubo(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    eta: float,
    penalty: float,
) -> tuple[np.ndarray, float]:
    """
    Build a full symmetric QUBO matrix.

    Convention:
        E(x) = x.T @ Q @ x + constant

    The cardinality constraint is enforced with:
        penalty * (sum(x) - K)^2
    """
    number_of_assets = len(
        INVESTABLE_ASSETS
    )

    k = NUMBER_OF_SELECTED_ASSETS

    qubo = (
        RISK_AVERSION
        / (k ** 2)
        * covariance.copy()
    )

    linear_terms = (
        - RETURN_WEIGHT
        / k
        * expected_returns
        + eta
        / k
        * systemic_risk
    )

    qubo[
        np.diag_indices(number_of_assets)
    ] += linear_terms

    # Expand:
    # A * (sum(x) - K)^2
    #
    # = A * [
    #     (1 - 2K) * sum(x_i)
    #     + 2 * sum_{i<j}(x_i x_j)
    #     + K^2
    # ]
    #
    # For a full symmetric Q matrix:
    # diagonal contribution = A * (1 - 2K)
    # each off-diagonal entry = A
    # constant = A * K^2

    diagonal_penalty = (
        penalty
        * (1 - 2 * k)
    )

    qubo[
        np.diag_indices(number_of_assets)
    ] += diagonal_penalty

    for i in range(number_of_assets):
        for j in range(
            i + 1,
            number_of_assets,
        ):
            qubo[i, j] += penalty
            qubo[j, i] += penalty

    constant = (
        penalty
        * (k ** 2)
    )

    return qubo, constant


def qubo_energy(
    x: np.ndarray,
    qubo: np.ndarray,
    constant: float,
) -> float:
    """
    Evaluate QUBO energy.
    """
    return float(
        x @ qubo @ x
        + constant
    )


def selected_assets_from_binary(
    x: np.ndarray,
) -> str:
    """
    Convert a binary vector into asset names.
    """
    selected = [
        asset
        for asset, value in zip(
            INVESTABLE_ASSETS,
            x,
        )
        if value == 1
    ]

    if not selected:
        return "None"

    return ", ".join(selected)


def enumerate_all_states(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    eta: float,
    penalty: float,
) -> pd.DataFrame:
    """
    Enumerate all 2^6 binary states.
    """
    qubo, constant = build_qubo(
        expected_returns,
        covariance,
        systemic_risk,
        eta,
        penalty,
    )

    rows = []

    for bits in product(
        [0, 1],
        repeat=len(INVESTABLE_ASSETS),
    ):
        x = np.array(
            bits,
            dtype=float,
        )

        cardinality = int(
            x.sum()
        )

        base_objective = original_objective(
            x,
            expected_returns,
            covariance,
            systemic_risk,
            eta,
        )

        penalty_value = (
            penalty
            * (
                cardinality
                - NUMBER_OF_SELECTED_ASSETS
            ) ** 2
        )

        energy = qubo_energy(
            x,
            qubo,
            constant,
        )

        rows.append(
            {
                "bitstring": "".join(
                    str(int(bit))
                    for bit in bits
                ),
                "selected_assets":
                    selected_assets_from_binary(x),
                "cardinality":
                    cardinality,
                "feasible":
                    cardinality
                    == NUMBER_OF_SELECTED_ASSETS,
                "original_objective":
                    base_objective,
                "penalty_value":
                    penalty_value,
                "qubo_energy":
                    energy,
                "energy_check_error":
                    abs(
                        energy
                        - (
                            base_objective
                            + penalty_value
                        )
                    ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "qubo_energy",
            ascending=True,
        )
        .reset_index(drop=True)
    )


def exact_feasible_optimum(
    states: pd.DataFrame,
) -> pd.Series:
    """
    Return the best feasible three-asset state.
    """
    feasible = states[
        states["feasible"]
    ]

    return (
        feasible.sort_values(
            "original_objective",
            ascending=True,
        )
        .iloc[0]
    )


def calibrate_penalty(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    eta: float,
) -> pd.DataFrame:
    """
    Test candidate penalty values against
    all binary states.
    """
    rows = []

    for penalty in PENALTY_VALUES:
        states = enumerate_all_states(
            expected_returns,
            covariance,
            systemic_risk,
            eta,
            penalty,
        )

        qubo_optimum = states.iloc[0]

        exact_optimum = (
            exact_feasible_optimum(
                states
            )
        )

        matches_exact = (
            qubo_optimum["bitstring"]
            == exact_optimum["bitstring"]
        )

        rows.append(
            {
                "eta": eta,
                "penalty":
                    penalty,
                "qubo_bitstring":
                    qubo_optimum[
                        "bitstring"
                    ],
                "qubo_selected_assets":
                    qubo_optimum[
                        "selected_assets"
                    ],
                "qubo_cardinality":
                    int(
                        qubo_optimum[
                            "cardinality"
                        ]
                    ),
                "qubo_feasible":
                    bool(
                        qubo_optimum[
                            "feasible"
                        ]
                    ),
                "exact_bitstring":
                    exact_optimum[
                        "bitstring"
                    ],
                "exact_selected_assets":
                    exact_optimum[
                        "selected_assets"
                    ],
                "matches_exact":
                    matches_exact,
                "qubo_energy":
                    qubo_optimum[
                        "qubo_energy"
                    ],
                "max_energy_check_error":
                    states[
                        "energy_check_error"
                    ].max(),
            }
        )

    return pd.DataFrame(
        rows
    )


def find_first_valid_penalty(
    calibration: pd.DataFrame,
) -> float:
    """
    Find the first tested penalty value
    producing the exact feasible optimum.
    """
    valid = calibration[
        calibration["qubo_feasible"]
        & calibration["matches_exact"]
    ]

    if valid.empty:
        raise ValueError(
            "No tested penalty value "
            "produced the exact feasible optimum."
        )

    return float(
        valid.iloc[0]["penalty"]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        expected_returns,
        covariance,
        systemic_risk,
    ) = prepare_inputs()

    expected_returns_array = (
        expected_returns.to_numpy()
    )

    covariance_array = (
        covariance.to_numpy()
    )

    systemic_risk_array = (
        systemic_risk.to_numpy()
    )

    standard_calibration = (
        calibrate_penalty(
            expected_returns_array,
            covariance_array,
            systemic_risk_array,
            STANDARD_ETA,
        )
    )

    systemic_calibration = (
        calibrate_penalty(
            expected_returns_array,
            covariance_array,
            systemic_risk_array,
            SYSTEMIC_ETA,
        )
    )

    combined_calibration = pd.concat(
        [
            standard_calibration,
            systemic_calibration,
        ],
        ignore_index=True,
    )

    standard_min_penalty = (
        find_first_valid_penalty(
            standard_calibration
        )
    )

    systemic_min_penalty = (
        find_first_valid_penalty(
            systemic_calibration
        )
    )

    common_min_penalty = max(
        standard_min_penalty,
        systemic_min_penalty,
    )

    standard_qubo, standard_constant = (
        build_qubo(
            expected_returns_array,
            covariance_array,
            systemic_risk_array,
            STANDARD_ETA,
            common_min_penalty,
        )
    )

    systemic_qubo, systemic_constant = (
        build_qubo(
            expected_returns_array,
            covariance_array,
            systemic_risk_array,
            SYSTEMIC_ETA,
            common_min_penalty,
        )
    )

    standard_states = enumerate_all_states(
        expected_returns_array,
        covariance_array,
        systemic_risk_array,
        STANDARD_ETA,
        common_min_penalty,
    )

    systemic_states = enumerate_all_states(
        expected_returns_array,
        covariance_array,
        systemic_risk_array,
        SYSTEMIC_ETA,
        common_min_penalty,
    )

    standard_matrix = pd.DataFrame(
        standard_qubo,
        index=INVESTABLE_ASSETS,
        columns=INVESTABLE_ASSETS,
    )

    systemic_matrix = pd.DataFrame(
        systemic_qubo,
        index=INVESTABLE_ASSETS,
        columns=INVESTABLE_ASSETS,
    )

    combined_calibration.to_csv(
        OUTPUT_DIR
        / "penalty_calibration.csv",
        index=False,
    )

    standard_matrix.to_csv(
        OUTPUT_DIR
        / "standard_qubo_matrix.csv"
    )

    systemic_matrix.to_csv(
        OUTPUT_DIR
        / "systemic_qubo_matrix.csv"
    )

    standard_states.to_csv(
        OUTPUT_DIR
        / "standard_qubo_states.csv",
        index=False,
    )

    systemic_states.to_csv(
        OUTPUT_DIR
        / "systemic_qubo_states.csv",
        index=False,
    )

    constants = pd.DataFrame(
        [
            {
                "model": "standard",
                "eta": STANDARD_ETA,
                "penalty":
                    common_min_penalty,
                "constant":
                    standard_constant,
            },
            {
                "model":
                    "systemic_risk_aware",
                "eta": SYSTEMIC_ETA,
                "penalty":
                    common_min_penalty,
                "constant":
                    systemic_constant,
            },
        ]
    )

    constants.to_csv(
        OUTPUT_DIR
        / "qubo_constants.csv",
        index=False,
    )

    print(
        "EPIC30 QUBO Formulation and "
        "Penalty Calibration"
    )
    print("-" * 72)

    print(
        "\nQUBO convention:"
    )
    print(
        "E(x) = x.T @ Q @ x + constant"
    )
    print(
        "Q representation: full symmetric matrix"
    )

    print(
        "\nBinary variables:",
        len(INVESTABLE_ASSETS),
    )
    print(
        "Total binary states:",
        2 ** len(INVESTABLE_ASSETS),
    )
    print(
        "Required cardinality:",
        NUMBER_OF_SELECTED_ASSETS,
    )

    print(
        "\nPenalty calibration"
    )
    print("-" * 72)

    for _, row in (
        combined_calibration.iterrows()
    ):
        print(
            f"eta={row['eta']:.2f} | "
            f"A={row['penalty']:.3f} | "
            f"k={int(row['qubo_cardinality'])} | "
            f"{row['qubo_selected_assets']:<30} | "
            f"feasible={row['qubo_feasible']} | "
            f"exact_match={row['matches_exact']}"
        )

    print(
        "\nFirst tested valid penalties"
    )
    print("-" * 72)

    print(
        "Standard eta=0.00:",
        f"A={standard_min_penalty:.3f}",
    )

    print(
        "Systemic eta=0.15:",
        f"A={systemic_min_penalty:.3f}",
    )

    print(
        "Common tested penalty:",
        f"A={common_min_penalty:.3f}",
    )

    standard_best = (
        standard_states.iloc[0]
    )

    systemic_best = (
        systemic_states.iloc[0]
    )

    print(
        "\nFinal QUBO validation"
    )
    print("-" * 72)

    print(
        "Standard:"
    )
    print(
        "  Selected:",
        standard_best[
            "selected_assets"
        ],
    )
    print(
        "  Cardinality:",
        int(
            standard_best[
                "cardinality"
            ]
        ),
    )
    print(
        "  QUBO energy:",
        f"{standard_best['qubo_energy']:.8f}",
    )

    print(
        "\nSystemic-aware:"
    )
    print(
        "  Selected:",
        systemic_best[
            "selected_assets"
        ],
    )
    print(
        "  Cardinality:",
        int(
            systemic_best[
                "cardinality"
            ]
        ),
    )
    print(
        "  QUBO energy:",
        f"{systemic_best['qubo_energy']:.8f}",
    )

    maximum_error = max(
        standard_states[
            "energy_check_error"
        ].max(),
        systemic_states[
            "energy_check_error"
        ].max(),
    )

    print(
        "\nMaximum formulation error:",
        f"{maximum_error:.3e}",
    )

    print(
        "\nSaved results:"
    )
    print(
        OUTPUT_DIR
        / "penalty_calibration.csv"
    )
    print(
        OUTPUT_DIR
        / "standard_qubo_matrix.csv"
    )
    print(
        OUTPUT_DIR
        / "systemic_qubo_matrix.csv"
    )
    print(
        OUTPUT_DIR
        / "standard_qubo_states.csv"
    )
    print(
        OUTPUT_DIR
        / "systemic_qubo_states.csv"
    )
    print(
        OUTPUT_DIR
        / "qubo_constants.csv"
    )


if __name__ == "__main__":
    main()