"""
EPIC30 - Financial Risk & Quantum Optimization

Comparison of alternative ETH factor models.

The purpose of this experiment is to test whether the simultaneous
inclusion of S&P 500 and NASDAQ creates multicollinearity that weakens
the interpretation of their individual effects.

Models:
    Model A:
        ETH ~ BTC + SP500 + GOLD + USD + VIX

    Model B:
        ETH ~ BTC + NASDAQ + GOLD + USD + VIX

    Model C:
        ETH ~ BTC + SP500 + NASDAQ + GOLD + USD + VIX

For each model, the script reports:
1. R-squared
2. Adjusted R-squared
3. OLS coefficients and p-values
4. HAC / Newey-West robust p-values
5. Variance Inflation Factors (VIF)
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


TARGET = "ETH"

MODELS = {
    "Model A - SP500": [
        "BTC",
        "SP500",
        "GOLD",
        "USD",
        "VIX",
    ],
    "Model B - NASDAQ": [
        "BTC",
        "NASDAQ",
        "GOLD",
        "USD",
        "VIX",
    ],
    "Model C - SP500 + NASDAQ": [
        "BTC",
        "SP500",
        "NASDAQ",
        "GOLD",
        "USD",
        "VIX",
    ],
}


def load_return_matrix(
    filename: str = "market_returns.csv",
) -> pd.DataFrame:
    """
    Load the return matrix created by data_loader.py.
    """

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(
            f"Return matrix not found: {data_path}\n"
            "Run src/data_loader.py first."
        )

    returns = pd.read_csv(
        data_path,
        index_col=0,
        parse_dates=True,
    )

    return returns


def fit_ols_model(
    returns: pd.DataFrame,
    factors: list[str],
):
    """
    Fit a standard OLS regression.
    """

    data = returns[[TARGET] + factors].dropna()

    y = data[TARGET]
    X = sm.add_constant(data[factors])

    model = sm.OLS(y, X).fit()

    return model, data


def fit_hac_model(
    returns: pd.DataFrame,
    factors: list[str],
    maxlags: int = 5,
):
    """
    Fit the same regression using Newey-West / HAC robust
    covariance estimation.

    Coefficient estimates remain the same as OLS, while
    standard errors and p-values are adjusted for possible
    heteroskedasticity and serial correlation.
    """

    data = returns[[TARGET] + factors].dropna()

    y = data[TARGET]
    X = sm.add_constant(data[factors])

    model = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )

    return model


def calculate_vif(
    data: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    """
    Calculate VIF for the explanatory variables.
    """

    X = data[factors].copy()

    vif = pd.DataFrame(
        {
            "factor": X.columns,
            "VIF": [
                variance_inflation_factor(
                    X.values,
                    i,
                )
                for i in range(X.shape[1])
            ],
        }
    )

    return (
        vif.sort_values(
            "VIF",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_coefficient_table(
    ols_model,
    hac_model,
) -> pd.DataFrame:
    """
    Compare standard OLS inference with HAC robust inference.
    """

    table = pd.DataFrame(
        {
            "coefficient": ols_model.params,
            "OLS_std_error": ols_model.bse,
            "OLS_p_value": ols_model.pvalues,
            "HAC_std_error": hac_model.bse,
            "HAC_p_value": hac_model.pvalues,
        }
    )

    return table


def run_model_comparison(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all configured models and return summary statistics
    together with detailed model outputs.
    """

    summary_rows = []
    model_outputs = {}

    for model_name, factors in MODELS.items():

        ols_model, data = fit_ols_model(
            returns,
            factors,
        )

        hac_model = fit_hac_model(
            returns,
            factors,
        )

        vif = calculate_vif(
            data,
            factors,
        )

        coefficients = build_coefficient_table(
            ols_model,
            hac_model,
        )

        summary_rows.append(
            {
                "model": model_name,
                "n_observations": int(ols_model.nobs),
                "r_squared": ols_model.rsquared,
                "adjusted_r_squared": ols_model.rsquared_adj,
                "max_vif": vif["VIF"].max(),
                "aic": ols_model.aic,
                "bic": ols_model.bic,
            }
        )

        model_outputs[model_name] = {
            "ols": ols_model,
            "hac": hac_model,
            "coefficients": coefficients,
            "vif": vif,
        }

    summary = pd.DataFrame(summary_rows)

    return summary, model_outputs


def save_results(
    summary: pd.DataFrame,
    model_outputs: dict,
) -> None:
    """
    Save model-comparison results to the data directory.
    """

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "model_comparison"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_dir / "model_summary.csv",
        index=False,
    )

    for model_name, outputs in model_outputs.items():

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("+", "plus")
            .replace("-", "")
        )

        outputs["coefficients"].to_csv(
            output_dir / f"{safe_name}_coefficients.csv"
        )

        outputs["vif"].to_csv(
            output_dir / f"{safe_name}_vif.csv",
            index=False,
        )


if __name__ == "__main__":

    returns = load_return_matrix()

    summary, model_outputs = run_model_comparison(
        returns
    )

    print("\nEPIC30 ETH Factor Model Comparison")
    print("----------------------------------")

    print("\nModel Summary")
    print("-------------")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    for model_name, outputs in model_outputs.items():

        print("\n")
        print("=" * 70)
        print(model_name)
        print("=" * 70)

        print("\nCoefficients and Inference")
        print("--------------------------")
        print(outputs["coefficients"])

        print("\nVariance Inflation Factors")
        print("--------------------------")
        print(outputs["vif"])

    save_results(
        summary,
        model_outputs,
    )

    print(
        "\nResults saved to:"
        "\ndata/model_comparison/"
    )