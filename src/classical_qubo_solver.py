from pathlib import Path

import dimod
import neal
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_DIR = (
    PROJECT_ROOT
    / "data"
    / "qubo"
)

STANDARD_QUBO_PATH = (
    QUBO_DIR
    / "standard_qubo_matrix.csv"
)

SYSTEMIC_QUBO_PATH = (
    QUBO_DIR
    / "systemic_qubo_matrix.csv"
)

CONSTANTS_PATH = (
    QUBO_DIR
    / "qubo_constants.csv"
)

STANDARD_STATES_PATH = (
    QUBO_DIR
    / "standard_qubo_states.csv"
)

SYSTEMIC_STATES_PATH = (
    QUBO_DIR
    / "systemic_qubo_states.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "classical_qubo"
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
CARDINALITY_PENALTY = 0.022

NUM_READS = 1000
RANDOM_SEED = 42


def load_qubo_matrix(
    path: Path,
) -> pd.DataFrame:
    """
    Load a full symmetric QUBO matrix.
    """
    matrix = pd.read_csv(
        path,
        index_col=0,
    )

    matrix = matrix.loc[
        INVESTABLE_ASSETS,
        INVESTABLE_ASSETS,
    ]

    return matrix


def load_constant(
    path: Path,
    model: str,
) -> float:
    """
    Load the constant offset associated with a QUBO model.
    """
    constants = pd.read_csv(path)

    row = constants[
        constants["model"] == model
    ]

    if row.empty:
        raise ValueError(
            f"No QUBO constant found for model: {model}"
        )

    return float(
        row.iloc[0]["constant"]
    )


def load_exact_solution(
    path: Path,
) -> pd.Series:
    """
    Load the exact QUBO ground state from
    exhaustive 64-state validation.
    """
    states = pd.read_csv(path)

    states = states.sort_values(
        "qubo_energy",
        ascending=True,
    )

    return states.iloc[0]


def validate_qubo_matrix(
    matrix: pd.DataFrame,
) -> None:
    """
    Validate shape, labels, and symmetry of
    the stored full QUBO matrix.
    """
    expected_shape = (
        len(INVESTABLE_ASSETS),
        len(INVESTABLE_ASSETS),
    )

    if matrix.shape != expected_shape:
        raise ValueError(
            "Unexpected QUBO matrix shape: "
            f"{matrix.shape}"
        )

    if not np.allclose(
        matrix.to_numpy(),
        matrix.to_numpy().T,
    ):
        raise ValueError(
            "QUBO matrix is not symmetric."
        )


def symmetric_matrix_to_bqm(
    matrix: pd.DataFrame,
    constant: float,
) -> dimod.BinaryQuadraticModel:
    """
    Convert a full symmetric matrix Q from

        E(x) = x.T @ Q @ x + constant

    into a BinaryQuadraticModel.

    Diagonal entries become linear biases.

    Because the symmetric matrix counts each
    off-diagonal interaction twice, pairwise
    BQM biases are 2 * Q[i, j].
    """
    linear = {}
    quadratic = {}

    for i, asset in enumerate(
        INVESTABLE_ASSETS
    ):
        linear[asset] = float(
            matrix.iloc[i, i]
        )

    for i in range(
        len(INVESTABLE_ASSETS)
    ):
        for j in range(
            i + 1,
            len(INVESTABLE_ASSETS),
        ):
            quadratic[
                (
                    INVESTABLE_ASSETS[i],
                    INVESTABLE_ASSETS[j],
                )
            ] = float(
                2.0
                * matrix.iloc[i, j]
            )

    return dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        constant,
        dimod.BINARY,
    )


def selected_assets_from_sample(
    sample: dict,
) -> list[str]:
    """
    Return selected assets from a binary sample.
    """
    return [
        asset
        for asset in INVESTABLE_ASSETS
        if int(sample[asset]) == 1
    ]


def sample_to_bitstring(
    sample: dict,
) -> str:
    """
    Convert a binary sample into a bitstring
    using the fixed asset ordering.
    """
    return "".join(
        str(
            int(sample[asset])
        )
        for asset in INVESTABLE_ASSETS
    )


def direct_matrix_energy(
    sample: dict,
    matrix: pd.DataFrame,
    constant: float,
) -> float:
    """
    Evaluate the original full symmetric
    matrix representation directly.
    """
    x = np.array(
        [
            int(sample[asset])
            for asset in INVESTABLE_ASSETS
        ],
        dtype=float,
    )

    return float(
        x
        @ matrix.to_numpy()
        @ x
        + constant
    )


def solve_with_simulated_annealing(
    model_name: str,
    eta: float,
    matrix: pd.DataFrame,
    constant: float,
    exact_solution: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Solve one QUBO using simulated annealing
    and compare the result with the exact
    ground state.
    """
    bqm = symmetric_matrix_to_bqm(
        matrix,
        constant,
    )

    sampler = neal.SimulatedAnnealingSampler()

    sampleset = sampler.sample(
        bqm,
        num_reads=NUM_READS,
        seed=RANDOM_SEED,
    )

    best = sampleset.first

    best_sample = {
        asset: int(
            best.sample[asset]
        )
        for asset in INVESTABLE_ASSETS
    }

    selected_assets = (
        selected_assets_from_sample(
            best_sample
        )
    )

    bitstring = sample_to_bitstring(
        best_sample
    )

    cardinality = int(
        sum(best_sample.values())
    )

    feasible = (
        cardinality
        == NUMBER_OF_SELECTED_ASSETS
    )

    solver_energy = float(
        best.energy
    )

    matrix_energy = direct_matrix_energy(
        best_sample,
        matrix,
        constant,
    )

    exact_bitstring = str(
        exact_solution["bitstring"]
    ).zfill(
        len(INVESTABLE_ASSETS)
    )

    exact_energy = float(
        exact_solution[
            "qubo_energy"
        ]
    )

    optimality_gap = (
        solver_energy
        - exact_energy
    )

    exact_match = (
        bitstring
        == exact_bitstring
    )

    energy_check_error = abs(
        solver_energy
        - matrix_energy
    )

    best_occurrences = int(
        sum(
            record.num_occurrences
            for record in sampleset.data(
                fields=[
                    "sample",
                    "energy",
                    "num_occurrences",
                ]
            )
            if np.isclose(
                record.energy,
                solver_energy,
                atol=1e-12,
            )
        )
    )

    success_rate = (
        best_occurrences
        / NUM_READS
    )

    result = {
        "model": model_name,
        "eta": eta,
        "penalty":
            CARDINALITY_PENALTY,
        "num_reads":
            NUM_READS,
        "bitstring":
            bitstring,
        "selected_assets":
            ", ".join(
                selected_assets
            ),
        "cardinality":
            cardinality,
        "feasible":
            feasible,
        "solver_energy":
            solver_energy,
        "matrix_energy":
            matrix_energy,
        "exact_energy":
            exact_energy,
        "optimality_gap":
            optimality_gap,
        "exact_match":
            exact_match,
        "energy_check_error":
            energy_check_error,
        "best_occurrences":
            best_occurrences,
        "success_rate":
            success_rate,
    }

    sample_rows = []

    for record in sampleset.data(
        fields=[
            "sample",
            "energy",
            "num_occurrences",
        ],
        sorted_by="energy",
    ):
        sample = {
            asset: int(
                record.sample[asset]
            )
            for asset in INVESTABLE_ASSETS
        }

        sample_rows.append(
            {
                "model":
                    model_name,
                "bitstring":
                    sample_to_bitstring(
                        sample
                    ),
                "selected_assets":
                    ", ".join(
                        selected_assets_from_sample(
                            sample
                        )
                    ),
                "cardinality":
                    int(
                        sum(
                            sample.values()
                        )
                    ),
                "energy":
                    float(
                        record.energy
                    ),
                "num_occurrences":
                    int(
                        record.num_occurrences
                    ),
            }
        )

    sample_summary = (
        pd.DataFrame(sample_rows)
        .groupby(
            [
                "model",
                "bitstring",
                "selected_assets",
                "cardinality",
                "energy",
            ],
            as_index=False,
            dropna=False,
        )[
            "num_occurrences"
        ]
        .sum()
        .sort_values(
            [
                "model",
                "energy",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    sample_summary[
        "frequency"
    ] = (
        sample_summary[
            "num_occurrences"
        ]
        / NUM_READS
    )

    return (
        result,
        sample_summary,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    standard_matrix = (
        load_qubo_matrix(
            STANDARD_QUBO_PATH
        )
    )

    systemic_matrix = (
        load_qubo_matrix(
            SYSTEMIC_QUBO_PATH
        )
    )

    validate_qubo_matrix(
        standard_matrix
    )

    validate_qubo_matrix(
        systemic_matrix
    )

    standard_constant = load_constant(
        CONSTANTS_PATH,
        "standard",
    )

    systemic_constant = load_constant(
        CONSTANTS_PATH,
        "systemic_risk_aware",
    )

    standard_exact = (
        load_exact_solution(
            STANDARD_STATES_PATH
        )
    )

    systemic_exact = (
        load_exact_solution(
            SYSTEMIC_STATES_PATH
        )
    )

    standard_result, standard_samples = (
        solve_with_simulated_annealing(
            model_name="standard",
            eta=STANDARD_ETA,
            matrix=standard_matrix,
            constant=standard_constant,
            exact_solution=standard_exact,
        )
    )

    systemic_result, systemic_samples = (
        solve_with_simulated_annealing(
            model_name="systemic_risk_aware",
            eta=SYSTEMIC_ETA,
            matrix=systemic_matrix,
            constant=systemic_constant,
            exact_solution=systemic_exact,
        )
    )

    solver_results = pd.DataFrame(
        [
            standard_result,
            systemic_result,
        ]
    )

    sample_summary = pd.concat(
        [
            standard_samples,
            systemic_samples,
        ],
        ignore_index=True,
    )

    solver_results.to_csv(
        OUTPUT_DIR
        / "solver_results.csv",
        index=False,
    )

    sample_summary.to_csv(
        OUTPUT_DIR
        / "sample_summary.csv",
        index=False,
    )

    print(
        "EPIC30 Classical QUBO Solver"
    )
    print("-" * 72)

    print(
        "\nSolver: Simulated Annealing"
    )
    print(
        f"Number of reads: {NUM_READS}"
    )
    print(
        f"Random seed: {RANDOM_SEED}"
    )
    print(
        "QUBO convention: "
        "E(x) = x.T @ Q @ x + constant"
    )
    print(
        f"Cardinality penalty: "
        f"A={CARDINALITY_PENALTY:.3f}"
    )

    for result in [
        standard_result,
        systemic_result,
    ]:
        print(
            "\n"
            + result["model"]
            .replace(
                "_",
                " ",
            )
            .title()
        )
        print("-" * 72)

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
            "Cardinality:",
            result[
                "cardinality"
            ],
        )

        print(
            "Feasible:",
            result[
                "feasible"
            ],
        )

        print(
            "Solver energy:",
            f"{result['solver_energy']:.8f}",
        )

        print(
            "Exact energy:",
            f"{result['exact_energy']:.8f}",
        )

        print(
            "Optimality gap:",
            f"{result['optimality_gap']:.3e}",
        )

        print(
            "Exact match:",
            result[
                "exact_match"
            ],
        )

        print(
            "Energy check error:",
            f"{result['energy_check_error']:.3e}",
        )

        print(
            "Ground-state occurrences:",
            result[
                "best_occurrences"
            ],
            "/",
            NUM_READS,
        )

        print(
            "Ground-state frequency:",
            f"{result['success_rate']:.3%}",
        )

    print(
        "\nSaved results"
    )
    print("-" * 72)

    print(
        OUTPUT_DIR
        / "solver_results.csv"
    )

    print(
        OUTPUT_DIR
        / "sample_summary.csv"
    )


if __name__ == "__main__":
    main()