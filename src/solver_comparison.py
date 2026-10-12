from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ISING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ising"
)

CLASSICAL_QUBO_DIR = (
    PROJECT_ROOT
    / "data"
    / "classical_qubo"
)

QAOA_DIR = (
    PROJECT_ROOT
    / "data"
    / "qaoa"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "comparison"
)


EXACT_VALIDATION_PATH = (
    ISING_DIR
    / "energy_validation.csv"
)

SA_RESULTS_PATH = (
    CLASSICAL_QUBO_DIR
    / "solver_results.csv"
)

SA_SAMPLES_PATH = (
    CLASSICAL_QUBO_DIR
    / "sample_summary.csv"
)

QAOA_RESULTS_PATH = (
    QAOA_DIR
    / "qaoa_results.csv"
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

STANDARD_ETA = 0.0
SYSTEMIC_ETA = 0.15


def normalize_model_name(
    model: str,
) -> str:
    """
    Normalize model labels across result files.
    """
    if model == "standard":
        return "standard"

    if model in [
        "systemic",
        "systemic_risk_aware",
    ]:
        return "systemic"

    raise ValueError(
        f"Unexpected model name: {model}"
    )


def load_exact_references(
    path: Path,
) -> pd.DataFrame:
    """
    Load exact ground-state references from
    the validated Ising state enumeration.
    """
    validation = pd.read_csv(
        path,
        dtype={
            "bitstring": str,
        },
    )

    validation["model"] = (
        validation["model"]
        .apply(
            normalize_model_name
        )
    )

    references = []

    for model in [
        "standard",
        "systemic",
    ]:
        model_rows = validation[
            validation["model"]
            == model
        ]

        model_rows = (
            model_rows.sort_values(
                "ising_energy",
                ascending=True,
            )
        )

        best = model_rows.iloc[0]

        references.append(
            {
                "model":
                    model,
                "exact_bitstring":
                    str(
                        best[
                            "bitstring"
                        ]
                    ).zfill(
                        len(
                            INVESTABLE_ASSETS
                        )
                    ),
                "exact_selected_assets":
                    best[
                        "selected_assets"
                    ],
                "exact_energy":
                    float(
                        best[
                            "ising_energy"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        references
    ).set_index(
        "model"
    )


def load_sa_results(
    results_path: Path,
    samples_path: Path,
    exact_references: pd.DataFrame,
) -> list[dict]:
    """
    Load simulated-annealing results and
    calculate feasible sampling frequency.
    """
    results = pd.read_csv(
        results_path,
        dtype={
            "bitstring": str,
        },
    )

    samples = pd.read_csv(
        samples_path,
        dtype={
            "bitstring": str,
        },
    )

    results["model"] = (
        results["model"]
        .apply(
            normalize_model_name
        )
    )

    samples["model"] = (
        samples["model"]
        .apply(
            normalize_model_name
        )
    )

    rows = []

    for _, result in results.iterrows():
        model = result[
            "model"
        ]

        model_samples = samples[
            samples["model"]
            == model
        ]

        feasible_sampling_rate = float(
            model_samples.loc[
                model_samples[
                    "cardinality"
                ]
                == NUMBER_OF_SELECTED_ASSETS,
                "frequency",
            ].sum()
        )

        exact_reference = (
            exact_references.loc[
                model
            ]
        )

        rows.append(
            {
                "model":
                    model,
                "eta":
                    (
                        STANDARD_ETA
                        if model
                        == "standard"
                        else SYSTEMIC_ETA
                    ),
                "method":
                    "Simulated Annealing",
                "qaoa_reps":
                    np.nan,
                "selected_assets":
                    result[
                        "selected_assets"
                    ],
                "bitstring":
                    str(
                        result[
                            "bitstring"
                        ]
                    ).zfill(
                        len(
                            INVESTABLE_ASSETS
                        )
                    ),
                "cardinality":
                    int(
                        result[
                            "cardinality"
                        ]
                    ),
                "feasible":
                    bool(
                        result[
                            "feasible"
                        ]
                    ),
                "solution_energy":
                    float(
                        result[
                            "solver_energy"
                        ]
                    ),
                "exact_bitstring":
                    exact_reference[
                        "exact_bitstring"
                    ],
                "exact_energy":
                    float(
                        exact_reference[
                            "exact_energy"
                        ]
                    ),
                "optimality_gap":
                    float(
                        result[
                            "optimality_gap"
                        ]
                    ),
                "exact_state_observed":
                    bool(
                        result[
                            "exact_match"
                        ]
                    ),
                "ground_state_sampling_rate":
                    float(
                        result[
                            "success_rate"
                        ]
                    ),
                "sampling_metric_type":
                    "SA read frequency",
                "feasible_sampling_rate":
                    feasible_sampling_rate,
                "most_likely_bitstring":
                    np.nan,
                "most_likely_exact_match":
                    np.nan,
                "sampling_budget":
                    int(
                        result[
                            "num_reads"
                        ]
                    ),
                "random_seed":
                    42,
                "optimizer":
                    "Simulated Annealing",
                "optimizer_evaluations":
                    np.nan,
            }
        )

    return rows


def load_qaoa_results(
    path: Path,
    exact_references: pd.DataFrame,
) -> list[dict]:
    """
    Load QAOA p=1 and p=2 benchmark results.
    """
    results = pd.read_csv(
        path,
        dtype={
            "best_bitstring": str,
            "exact_bitstring": str,
            "most_likely_bitstring": str,
        },
    )

    results["model"] = (
        results["model"]
        .apply(
            normalize_model_name
        )
    )

    rows = []

    for _, result in results.iterrows():
        model = result[
            "model"
        ]

        exact_reference = (
            exact_references.loc[
                model
            ]
        )

        rows.append(
            {
                "model":
                    model,
                "eta":
                    float(
                        result[
                            "eta"
                        ]
                    ),
                "method":
                    "QAOA",
                "qaoa_reps":
                    int(
                        result[
                            "qaoa_reps"
                        ]
                    ),
                "selected_assets":
                    result[
                        "best_selected_assets"
                    ],
                "bitstring":
                    str(
                        result[
                            "best_bitstring"
                        ]
                    ).zfill(
                        len(
                            INVESTABLE_ASSETS
                        )
                    ),
                "cardinality":
                    int(
                        result[
                            "best_cardinality"
                        ]
                    ),
                "feasible":
                    bool(
                        result[
                            "best_feasible"
                        ]
                    ),
                "solution_energy":
                    float(
                        result[
                            "best_energy"
                        ]
                    ),
                "exact_bitstring":
                    exact_reference[
                        "exact_bitstring"
                    ],
                "exact_energy":
                    float(
                        exact_reference[
                            "exact_energy"
                        ]
                    ),
                "optimality_gap":
                    float(
                        result[
                            "optimality_gap"
                        ]
                    ),
                "exact_state_observed":
                    bool(
                        result[
                            "exact_match"
                        ]
                    ),
                "ground_state_sampling_rate":
                    float(
                        result[
                            "ground_state_probability"
                        ]
                    ),
                "sampling_metric_type":
                    "QAOA sample probability",
                "feasible_sampling_rate":
                    float(
                        result[
                            "feasible_probability"
                        ]
                    ),
                "most_likely_bitstring":
                    str(
                        result[
                            "most_likely_bitstring"
                        ]
                    ).zfill(
                        len(
                            INVESTABLE_ASSETS
                        )
                    ),
                "most_likely_exact_match":
                    bool(
                        result[
                            "most_likely_exact_match"
                        ]
                    ),
                "sampling_budget":
                    int(
                        result[
                            "shots"
                        ]
                    ),
                "random_seed":
                    int(
                        result[
                            "random_seed"
                        ]
                    ),
                "optimizer":
                    result[
                        "optimizer"
                    ],
                "optimizer_evaluations":
                    int(
                        result[
                            "optimizer_evaluations"
                        ]
                    ),
            }
        )

    return rows


def build_exact_rows(
    exact_references: pd.DataFrame,
) -> list[dict]:
    """
    Build exact-enumeration reference rows.
    """
    rows = []

    for model in [
        "standard",
        "systemic",
    ]:
        reference = (
            exact_references.loc[
                model
            ]
        )

        bitstring = str(
            reference[
                "exact_bitstring"
            ]
        )

        cardinality = sum(
            int(value)
            for value in bitstring
        )

        rows.append(
            {
                "model":
                    model,
                "eta":
                    (
                        STANDARD_ETA
                        if model
                        == "standard"
                        else SYSTEMIC_ETA
                    ),
                "method":
                    "Exact Enumeration",
                "qaoa_reps":
                    np.nan,
                "selected_assets":
                    reference[
                        "exact_selected_assets"
                    ],
                "bitstring":
                    bitstring,
                "cardinality":
                    cardinality,
                "feasible":
                    cardinality
                    == NUMBER_OF_SELECTED_ASSETS,
                "solution_energy":
                    float(
                        reference[
                            "exact_energy"
                        ]
                    ),
                "exact_bitstring":
                    bitstring,
                "exact_energy":
                    float(
                        reference[
                            "exact_energy"
                        ]
                    ),
                "optimality_gap":
                    0.0,
                "exact_state_observed":
                    True,
                "ground_state_sampling_rate":
                    np.nan,
                "sampling_metric_type":
                    "Exact reference",
                "feasible_sampling_rate":
                    np.nan,
                "most_likely_bitstring":
                    np.nan,
                "most_likely_exact_match":
                    np.nan,
                "sampling_budget":
                    np.nan,
                "random_seed":
                    np.nan,
                "optimizer":
                    "Exact Enumeration",
                "optimizer_evaluations":
                    np.nan,
            }
        )

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact_references = (
        load_exact_references(
            EXACT_VALIDATION_PATH
        )
    )

    exact_rows = build_exact_rows(
        exact_references
    )

    sa_rows = load_sa_results(
        results_path=SA_RESULTS_PATH,
        samples_path=SA_SAMPLES_PATH,
        exact_references=exact_references,
    )

    qaoa_rows = load_qaoa_results(
        path=QAOA_RESULTS_PATH,
        exact_references=exact_references,
    )

    comparison = pd.DataFrame(
        exact_rows
        + sa_rows
        + qaoa_rows
    )

    method_order = {
        "Exact Enumeration": 0,
        "Simulated Annealing": 1,
        "QAOA": 2,
    }

    model_order = {
        "standard": 0,
        "systemic": 1,
    }

    comparison[
        "_model_order"
    ] = comparison[
        "model"
    ].map(
        model_order
    )

    comparison[
        "_method_order"
    ] = comparison[
        "method"
    ].map(
        method_order
    )

    comparison = comparison.sort_values(
        [
            "_model_order",
            "_method_order",
            "qaoa_reps",
        ],
        na_position="first",
    ).drop(
        columns=[
            "_model_order",
            "_method_order",
        ]
    ).reset_index(
        drop=True
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "solver_comparison.csv",
        index=False,
    )

    print(
        "EPIC30 Solver Comparison"
    )
    print("-" * 72)

    print(
        "\nCompared methods:"
    )
    print(
        "Exact Enumeration"
    )
    print(
        "Simulated Annealing"
    )
    print(
        "QAOA p=1"
    )
    print(
        "QAOA p=2"
    )

    for model in [
        "standard",
        "systemic",
    ]:
        print(
            "\n"
            + model.title()
        )
        print("-" * 72)

        model_results = comparison[
            comparison["model"]
            == model
        ]

        for _, result in (
            model_results.iterrows()
        ):
            method = result[
                "method"
            ]

            if method == "QAOA":
                method = (
                    f"QAOA p="
                    f"{int(result['qaoa_reps'])}"
                )

            print(
                "\nMethod:",
                method,
            )

            print(
                "Selected assets:",
                result[
                    "selected_assets"
                ],
            )

            print(
                "Bitstring:",
                result[
                    "bitstring"
                ],
            )

            print(
                "Feasible:",
                result[
                    "feasible"
                ],
            )

            print(
                "Energy:",
                f"{result['solution_energy']:.8f}",
            )

            print(
                "Optimality gap:",
                f"{result['optimality_gap']:.3e}",
            )

            print(
                "Exact state observed:",
                result[
                    "exact_state_observed"
                ],
            )

            if not pd.isna(
                result[
                    "ground_state_sampling_rate"
                ]
            ):
                print(
                    "Ground-state sampling rate:",
                    f"{result['ground_state_sampling_rate']:.3%}",
                )

            if not pd.isna(
                result[
                    "feasible_sampling_rate"
                ]
            ):
                print(
                    "Feasible sampling rate:",
                    f"{result['feasible_sampling_rate']:.3%}",
                )

    print(
        "\nSaved results"
    )
    print("-" * 72)

    print(
        OUTPUT_DIR
        / "solver_comparison.csv"
    )


if __name__ == "__main__":
    main()