from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    / "classical_optimization"
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


RISK_AVERSION = 0.5
RETURN_WEIGHT = 0.5

SYSTEMIC_RISK_WEIGHTS = [
    # Baseline
    0.000,

    # Very low penalty
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.040,
    0.050,

    # Low-to-moderate penalty
    0.060,
    0.075,
    0.090,
    0.100,
    0.125,
    0.150,

    # Moderate penalty
    0.175,
    0.200,
    0.225,
    0.250,
    0.300,

    # Strong penalty
    0.350,
    0.400,
    0.500,

    # Stress / upper-range sensitivity
    0.600,
    0.750,
    1.000,
]

TRADING_DAYS = 252
COVARIANCE_WINDOW = 252
ACTIVE_WEIGHT_THRESHOLD = 1e-4


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

    returns = returns.select_dtypes(
        include=[np.number]
    )

    return returns


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
    Calculate portfolio exposure to the
    scaled systemic risk proxy.
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
    Mean-variance objective with an optional
    systemic risk penalty.
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


def optimize_portfolio(
    number_of_assets: int,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
    systemic_risk_weight: float,
):
    """
    Solve a long-only fully invested
    continuous portfolio problem.
    """
    initial_weights = np.full(
        number_of_assets,
        1.0 / number_of_assets,
    )

    bounds = [
        (0.0, 1.0)
        for _ in range(
            number_of_assets
        )
    ]

    constraints = {
        "type": "eq",
        "fun": lambda weights:
            np.sum(weights) - 1.0,
    }

    result = minimize(
        portfolio_objective,
        initial_weights,
        args=(
            expected_returns,
            covariance,
            systemic_risk,
            systemic_risk_weight,
        ),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(
            "Optimization failed for "
            f"systemic risk weight "
            f"{systemic_risk_weight:.3f}: "
            f"{result.message}"
        )

    return result


def get_active_assets(
    assets: list[str],
    weights: np.ndarray,
) -> list[str]:
    """
    Identify assets with economically non-zero
    portfolio weights.
    """
    return [
        asset
        for asset, weight in zip(
            assets,
            weights,
        )
        if weight > ACTIVE_WEIGHT_THRESHOLD
    ]


def create_summary_row(
    assets: list[str],
    systemic_risk_weight: float,
    weights: np.ndarray,
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    systemic_risk: np.ndarray,
) -> dict:
    """
    Create portfolio-level metrics for one
    systemic risk penalty setting.
    """
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

    objective_value = (
        portfolio_objective(
            weights,
            expected_returns,
            covariance,
            systemic_risk,
            systemic_risk_weight,
        )
    )

    active_assets = get_active_assets(
        assets,
        weights,
    )

    model = (
        "standard"
        if np.isclose(
            systemic_risk_weight,
            0.0,
        )
        else "systemic_risk_aware"
    )

    return {
        "model":
            model,
        "systemic_risk_weight":
            systemic_risk_weight,
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
        "active_assets":
            ", ".join(active_assets),
        "number_of_active_assets":
            len(active_assets),
    }


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

    input_table = pd.DataFrame(
        {
            "asset":
                assets,
            "expected_return":
                expected_returns.to_numpy(),
            "systemic_risk_score":
                systemic_risk.to_numpy(),
            "scaled_systemic_risk":
                scaled_systemic_risk.to_numpy(),
        }
    )

    weight_rows = []
    summary_rows = []

    for systemic_risk_weight in (
        SYSTEMIC_RISK_WEIGHTS
    ):
        result = optimize_portfolio(
            len(assets),
            expected_returns_array,
            covariance_array,
            systemic_risk_array,
            systemic_risk_weight,
        )

        model = (
            "standard"
            if np.isclose(
                systemic_risk_weight,
                0.0,
            )
            else "systemic_risk_aware"
        )

        active_assets = get_active_assets(
            assets,
            result.x,
        )

        weight_row = {
            "model":
                model,
            "systemic_risk_weight":
                systemic_risk_weight,
            "active_assets":
                ", ".join(active_assets),
            "number_of_active_assets":
                len(active_assets),
        }

        for asset, weight in zip(
            assets,
            result.x,
        ):
            weight_row[
                asset
            ] = weight

        weight_rows.append(
            weight_row
        )

        summary_rows.append(
            create_summary_row(
                assets,
                systemic_risk_weight,
                result.x,
                expected_returns_array,
                covariance_array,
                systemic_risk_array,
            )
        )

    sensitivity_weights = pd.DataFrame(
        weight_rows
    )

    portfolio_summary = pd.DataFrame(
        summary_rows
    )

    standard_weights = (
        sensitivity_weights.loc[
            sensitivity_weights[
                "model"
            ] == "standard"
        ].copy()
    )

    input_table.to_csv(
        OUTPUT_DIR
        / "optimization_inputs.csv",
        index=False,
    )

    standard_weights.to_csv(
        OUTPUT_DIR
        / "portfolio_weights.csv",
        index=False,
    )

    portfolio_summary.to_csv(
        OUTPUT_DIR
        / "portfolio_summary.csv",
        index=False,
    )

    sensitivity_weights.to_csv(
        OUTPUT_DIR
        / "systemic_risk_sensitivity.csv",
        index=False,
    )

    print(
        "EPIC30 Classical Portfolio Optimization"
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
        "from portfolio weights: "
        f"{', '.join(EXTERNAL_RISK_FACTORS)}"
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
        "Systemic risk weights: "
        + ", ".join(
            f"{weight:.3f}"
            for weight
            in SYSTEMIC_RISK_WEIGHTS
        )
    )

    print(
        "\nOptimization Inputs"
    )
    print("-" * 70)

    print(
        input_table.round(
            6
        ).to_string(
            index=False
        )
    )

    print(
        "\nPortfolio Weights by Systemic Risk Penalty"
    )
    print("-" * 70)

    print(
        sensitivity_weights.round(
            6
        ).to_string(
            index=False
        )
    )

    print(
        "\nPortfolio Summary"
    )
    print("-" * 70)

    print(
        portfolio_summary.round(
            6
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()