from itertools import combinations
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
    / "exact_discrete_optimization"
)


INVESTABLE_ASSETS = [
    "ETH",
    "BTC",
    "NASDAQ",
    "SP500",
    "GOLD",
    "USD",
]

EXTERNAL_RISK_FACTORS = [
    "VIX",
]

NUMBER_OF_SELECTED_ASSETS = 3

RISK_AVERSION = 0.5
RETURN_WEIGHT = 0.5

SYSTEMIC_RISK_WEIGHTS = [
    0.000,
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,
    0.060,
    0.075,
    0.090,
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
    0.225,
    0.250,
    0.300,
    0.350,
    0.400,
    0.500,
    0.600,
    0.750,
    1.000,
]

TRADING_DAYS = 252
COVARIANCE_WINDOW = 252


def load_returns(
    path: Path,
) -> pd.DataFrame:
    """
    Load market returns and retain numeric asset columns.
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


def validate_assets(
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    systemic_risk: pd.Series,
) -> None:
    """
    Confirm that all investable assets are available
    in the required input datasets.
    """
    missing_returns = [
        asset
        for asset in INVESTABLE_ASSETS
        if asset not in returns.columns
    ]

    missing_covariance = [
        asset
        for asset in INVESTABLE_ASSETS
        if asset not in covariance.columns
        or asset not in covariance.index
    ]

    missing_systemic_risk = [
        asset
        for asset in INVESTABLE_ASSETS
        if asset not in systemic_risk.index
    ]

    if missing_returns:
        raise ValueError(
            "Missing investable assets in returns: "
            f"{missing_returns}"
        )

    if missing_covariance:
        raise ValueError(
            "Missing investable assets in covariance: "
            f"{missing_covariance}"
        )

    if missing_systemic_risk:
        raise ValueError(
            "Missing investable assets in systemic risk scores: "
            f"{missing_systemic_risk}"
        )


def calculate_expected_returns(
    returns: pd.DataFrame,
    assets: list[str],
) -> pd.Series:
    """
    Calculate annualized long-run historical mean returns
    using the full available return sample.
    """
    return (
        returns[assets].mean()
        * TRADING_DAYS
    )


def annualize_covariance(
    covariance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Annualize the daily covariance matrix.
    """
    return (
        covariance
        * TRADING_DAYS
    )


def scale_systemic_risk(
    systemic_risk: pd.Series,
    assets: list[str],
) -> pd.Series:
    """
    Scale systemic risk scores relative to the
    maximum score among investable assets.
    """
    scores = systemic_risk.reindex(
        assets
    )

    maximum = float(
        scores.max()
    )

    if maximum <= 0:
        raise ValueError(
            "Maximum systemic risk score "
            "must be positive."
        )

    return (
        scores
        / maximum
    )


def create_binary_vector(
    assets: list[str],
    selected_assets: tuple[str, ...],
) -> np.ndarray:
    """
    Create a binary selection vector.
    """
    return np.array(
        [
            1.0
            if asset in selected_assets
            else 0.0
            for asset in assets
        ],
        dtype=float,
    )


def create_portfolio_weights(
    binary_vector: np.ndarray,
) -> np.ndarray:
    """
    Convert a binary selection vector into
    an equal-weight portfolio.
    """
    return (
        binary_vector
        / NUMBER_OF_SELECTED_ASSETS
    )


def portfolio_return(
    weights: np.ndarray,
    expected_returns: np.ndarray,
) -> float:
    """
    Calculate expected portfolio return.
    """
    return float(
        weights @ expected_returns
    )


def portfolio_variance(
    weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    """
    Calculate portfolio variance.
    """
    return float(
        weights
        @ covariance
        @ weights
    )


def portfolio_systemic_risk(
    weights: np.ndarray,
    systemic_risk: np.ndarray,
) -> float:
    """
    Calculate portfolio systemic-risk exposure.
    """
    return float(
        weights @ systemic_risk
    )


def portfolio_objective(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    systemic_risk_weight: float,
) -> float:
    """
    Evaluate the discrete portfolio objective.
    """
    risk = portfolio_variance(
        weights,
        covariance,
    )

    expected_return = portfolio_return(
        weights,
        expected_returns,
    )

    systemic_exposure = (
        portfolio_systemic_risk(
            weights,
            systemic_risk,
        )
    )

    return (
        RISK_AVERSION * risk
        - RETURN_WEIGHT * expected_return
        + systemic_risk_weight
        * systemic_exposure
    )


def evaluate_portfolio(
    assets: list[str],
    selected_assets: tuple[str, ...],
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    systemic_risk_weight: float,
) -> dict:
    """
    Evaluate one discrete equal-weight portfolio.
    """
    binary_vector = create_binary_vector(
        assets,
        selected_assets,
    )

    weights = create_portfolio_weights(
        binary_vector
    )

    expected_return = portfolio_return(
        weights,
        expected_returns,
    )

    variance = portfolio_variance(
        weights,
        covariance,
    )

    volatility = float(
        np.sqrt(variance)
    )

    systemic_exposure = (
        portfolio_systemic_risk(
            weights,
            systemic_risk,
        )
    )

    objective_value = portfolio_objective(
        weights,
        expected_returns,
        covariance,
        systemic_risk,
        systemic_risk_weight,
    )

    row = {
        "model": (
            "standard"
            if np.isclose(
                systemic_risk_weight,
                0.0,
            )
            else "systemic_risk_aware"
        ),
        "systemic_risk_weight":
            systemic_risk_weight,
        "selected_assets":
            ", ".join(selected_assets),
        "expected_return":
            expected_return,
        "variance":
            variance,
        "volatility":
            volatility,
        "systemic_risk_exposure":
            systemic_exposure,
        "objective_value":
            objective_value,
    }

    for asset, selected, weight in zip(
        assets,
        binary_vector,
        weights,
    ):
        row[
            f"{asset}_selected"
        ] = int(selected)

        row[
            f"{asset}_weight"
        ] = weight

    return row


def enumerate_portfolios(
    assets: list[str],
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    systemic_risk_weight: float,
) -> pd.DataFrame:
    """
    Enumerate all feasible three-asset portfolios.
    """
    rows = []

    for selected_assets in combinations(
        assets,
        NUMBER_OF_SELECTED_ASSETS,
    ):
        row = evaluate_portfolio(
            assets,
            selected_assets,
            expected_returns,
            covariance,
            systemic_risk,
            systemic_risk_weight,
        )

        rows.append(row)

    results = pd.DataFrame(
        rows
    )

    return results.sort_values(
        "objective_value",
        ascending=True,
    ).reset_index(
        drop=True
    )


def find_optimal_portfolio(
    results: pd.DataFrame,
) -> pd.Series:
    """
    Return the globally optimal portfolio
    from exhaustive enumeration.
    """
    return results.iloc[0]


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    returns = load_returns(
        RETURNS_PATH
    )

    full_covariance = load_covariance(
        COVARIANCE_PATH
    )

    full_systemic_risk = load_systemic_risk(
        SYSTEMIC_RISK_PATH
    )

    validate_assets(
        returns,
        full_covariance,
        full_systemic_risk,
    )

    assets = INVESTABLE_ASSETS

    covariance = (
        full_covariance.loc[
            assets,
            assets,
        ]
    )

    systemic_risk = (
        full_systemic_risk.reindex(
            assets
        )
    )

    expected_returns = (
        calculate_expected_returns(
            returns,
            assets,
        )
    )

    annual_covariance = (
        annualize_covariance(
            covariance
        )
    )

    scaled_systemic_risk = (
        scale_systemic_risk(
            systemic_risk,
            assets,
        )
    )

    expected_returns_array = (
        expected_returns.to_numpy()
    )

    covariance_array = (
        annual_covariance.to_numpy()
    )

    systemic_risk_array = (
        scaled_systemic_risk.to_numpy()
    )

    standard_results = enumerate_portfolios(
        assets,
        expected_returns_array,
        covariance_array,
        systemic_risk_array,
        systemic_risk_weight=0.0,
    )

    standard_optimum = find_optimal_portfolio(
        standard_results
    )

    sensitivity_rows = []

    previous_portfolio = None

    print(
        "EPIC30 Exact Discrete Portfolio Optimization"
    )
    print("-" * 70)

    print(
        f"\nNetwork nodes: "
        f"{len(full_systemic_risk)}"
    )

    print(
        f"Investable assets: "
        f"{len(assets)}"
    )

    print(
        "External risk factors excluded "
        "from portfolio selection: "
        f"{', '.join(EXTERNAL_RISK_FACTORS)}"
    )

    print(
        f"\nSelected assets per portfolio: "
        f"{NUMBER_OF_SELECTED_ASSETS}"
    )

    print(
        f"Equal weight per selected asset: "
        f"{1 / NUMBER_OF_SELECTED_ASSETS:.6f}"
    )

    print(
        f"Feasible portfolios: "
        f"{len(standard_results)}"
    )

    print(
        "\nExpected return estimation: "
        "full available sample"
    )

    print(
        f"Expected return observations: "
        f"{len(returns)}"
    )

    print(
        f"Covariance estimation: "
        f"latest {COVARIANCE_WINDOW} observations"
    )

    print(
        f"Systemic risk estimation: "
        f"latest {COVARIANCE_WINDOW}-observation "
        "dependence network"
    )

    print(
        f"\nRisk aversion: "
        f"{RISK_AVERSION:.2f}"
    )

    print(
        f"Return weight: "
        f"{RETURN_WEIGHT:.2f}"
    )

    print(
        f"Systemic risk sweep points: "
        f"{len(SYSTEMIC_RISK_WEIGHTS)}"
    )

    print(
        "\nStandard Exact Optimum"
    )
    print("-" * 70)

    print(
        f"Selected assets: "
        f"{standard_optimum['selected_assets']}"
    )

    print(
        f"Expected return: "
        f"{standard_optimum['expected_return']:.6f}"
    )

    print(
        f"Variance: "
        f"{standard_optimum['variance']:.6f}"
    )

    print(
        f"Volatility: "
        f"{standard_optimum['volatility']:.6f}"
    )

    print(
        "Systemic risk exposure: "
        f"{standard_optimum['systemic_risk_exposure']:.6f}"
    )

    print(
        f"Objective value: "
        f"{standard_optimum['objective_value']:.6f}"
    )

    print(
        "\nSystemic-Risk Sensitivity"
    )
    print("-" * 70)

    for systemic_risk_weight in (
        SYSTEMIC_RISK_WEIGHTS
    ):
        results = enumerate_portfolios(
            assets,
            expected_returns_array,
            covariance_array,
            systemic_risk_array,
            systemic_risk_weight,
        )

        optimum = find_optimal_portfolio(
            results
        )

        sensitivity_rows.append(
            optimum.to_dict()
        )

        selected_portfolio = optimum[
            "selected_assets"
        ]

        transition_marker = ""

        if (
            previous_portfolio is None
            or selected_portfolio
            != previous_portfolio
        ):
            transition_marker = "  <-- transition"

        print(
            f"eta = "
            f"{systemic_risk_weight:>5.3f} | "
            f"{selected_portfolio:<25} | "
            f"return = "
            f"{optimum['expected_return']:.6f} | "
            f"vol = "
            f"{optimum['volatility']:.6f} | "
            f"systemic = "
            f"{optimum['systemic_risk_exposure']:.6f}"
            f"{transition_marker}"
        )

        previous_portfolio = (
            selected_portfolio
        )

    sensitivity_results = pd.DataFrame(
        sensitivity_rows
    )

    standard_results.to_csv(
        OUTPUT_DIR
        / "standard_portfolios.csv",
        index=False,
    )

    pd.DataFrame(
        [
            standard_optimum.to_dict()
        ]
    ).to_csv(
        OUTPUT_DIR
        / "optimal_portfolio.csv",
        index=False,
    )

    sensitivity_results.to_csv(
        OUTPUT_DIR
        / "systemic_risk_sensitivity.csv",
        index=False,
    )

    print(
        "\nSaved Results"
    )
    print("-" * 70)

    print(
        OUTPUT_DIR
        / "standard_portfolios.csv"
    )

    print(
        OUTPUT_DIR
        / "optimal_portfolio.csv"
    )

    print(
        OUTPUT_DIR
        / "systemic_risk_sensitivity.csv"
    )


if __name__ == "__main__":
    main()