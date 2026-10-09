from pathlib import Path

import numpy as np
import pandas as pd

from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ISING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ising"
)

STANDARD_LINEAR_PATH = (
    ISING_DIR
    / "standard_ising_linear.csv"
)

STANDARD_QUADRATIC_PATH = (
    ISING_DIR
    / "standard_ising_quadratic.csv"
)

SYSTEMIC_LINEAR_PATH = (
    ISING_DIR
    / "systemic_ising_linear.csv"
)

SYSTEMIC_QUADRATIC_PATH = (
    ISING_DIR
    / "systemic_ising_quadratic.csv"
)

CONSTANTS_PATH = (
    ISING_DIR
    / "ising_constants.csv"
)

VALIDATION_PATH = (
    ISING_DIR
    / "energy_validation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "qaoa"
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

QAOA_REPS_VALUES = [
    1,
    2,
]

SHOTS = 1000
MAX_ITERATIONS = 200
RANDOM_SEED = 42
INITIAL_POINT_VALUE = 0.5


def load_ising_coefficients(
    linear_path: Path,
    quadratic_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Ising linear and quadratic coefficients.
    """
    linear = pd.read_csv(
        linear_path
    )

    quadratic = pd.read_csv(
        quadratic_path
    )

    linear = linear.set_index(
        "asset"
    ).loc[
        INVESTABLE_ASSETS
    ].reset_index()

    return (
        linear,
        quadratic,
    )


def load_ising_offset(
    path: Path,
    model: str,
) -> tuple[float, float, float]:
    """
    Load penalty, QUBO constant, and Ising offset
    associated with one model.
    """
    constants = pd.read_csv(path)

    row = constants[
        constants["model"] == model
    ]

    if row.empty:
        raise ValueError(
            f"No Ising metadata found for model: {model}"
        )

    penalty = float(
        row.iloc[0]["penalty"]
    )

    qubo_constant = float(
        row.iloc[0][
            "qubo_constant"
        ]
    )

    ising_offset = float(
        row.iloc[0][
            "ising_offset"
        ]
    )

    return (
        penalty,
        qubo_constant,
        ising_offset,
    )


def load_exact_solution(
    path: Path,
    model: str,
) -> pd.Series:
    """
    Load the exact Ising ground state from
    the 64-state energy validation.
    """
    validation = pd.read_csv(
        path,
        dtype={
            "bitstring": str,
        },
    )

    validation = validation[
        validation["model"] == model
    ]

    validation = validation.sort_values(
        "ising_energy",
        ascending=True,
    )

    return validation.iloc[0]


def pauli_label(
    z_qubits: list[int],
) -> str:
    """
    Build a Qiskit Pauli label.

    Asset index i is mapped to qubit i.
    Qiskit Pauli strings display qubit 0
    at the rightmost position.
    """
    label = [
        "I"
        for _ in INVESTABLE_ASSETS
    ]

    for qubit in z_qubits:
        label[
            len(INVESTABLE_ASSETS)
            - 1
            - qubit
        ] = "Z"

    return "".join(label)


def build_cost_hamiltonian(
    linear: pd.DataFrame,
    quadratic: pd.DataFrame,
    offset: float,
) -> SparsePauliOp:
    """
    Build the Ising cost Hamiltonian

        H = offset I
          + sum_i h_i Z_i
          + sum_{i<j} J_ij Z_i Z_j.
    """
    pauli_terms = []

    identity = (
        "I"
        * len(
            INVESTABLE_ASSETS
        )
    )

    pauli_terms.append(
        (
            identity,
            offset,
        )
    )

    for i, asset in enumerate(
        INVESTABLE_ASSETS
    ):
        coefficient = float(
            linear.loc[
                linear["asset"]
                == asset,
                "h",
            ].iloc[0]
        )

        pauli_terms.append(
            (
                pauli_label(
                    [i]
                ),
                coefficient,
            )
        )

    for _, row in quadratic.iterrows():
        asset_i = row[
            "asset_i"
        ]

        asset_j = row[
            "asset_j"
        ]

        i = INVESTABLE_ASSETS.index(
            asset_i
        )

        j = INVESTABLE_ASSETS.index(
            asset_j
        )

        coefficient = float(
            row["J"]
        )

        pauli_terms.append(
            (
                pauli_label(
                    [
                        i,
                        j,
                    ]
                ),
                coefficient,
            )
        )

    return SparsePauliOp.from_list(
        pauli_terms
    ).simplify()


def qiskit_state_to_bitstring(
    state,
) -> str:
    """
    Convert a Qiskit basis-state label into
    the fixed asset ordering

        ETH, BTC, NASDAQ, SP500, GOLD, USD.

    Qiskit displays basis strings as
    q_(n-1) ... q_0, so the string is reversed.
    """
    if isinstance(
        state,
        (int, np.integer),
    ):
        qiskit_bitstring = format(
            int(state),
            f"0{len(INVESTABLE_ASSETS)}b",
        )
    else:
        qiskit_bitstring = str(
            state
        ).replace(
            " ",
            "",
        )

        qiskit_bitstring = (
            qiskit_bitstring.zfill(
                len(
                    INVESTABLE_ASSETS
                )
            )
        )

    return qiskit_bitstring[
        ::-1
    ]


def bitstring_to_binary(
    bitstring: str,
) -> np.ndarray:
    """
    Convert an asset-order bitstring into
    a binary vector.
    """
    return np.array(
        [
            int(value)
            for value in bitstring
        ],
        dtype=int,
    )


def selected_assets_from_bitstring(
    bitstring: str,
) -> list[str]:
    """
    Return selected assets from an
    asset-order bitstring.
    """
    return [
        asset
        for asset, value in zip(
            INVESTABLE_ASSETS,
            bitstring,
        )
        if int(value) == 1
    ]


def calculate_ising_energy(
    bitstring: str,
    linear: pd.DataFrame,
    quadratic: pd.DataFrame,
    offset: float,
) -> float:
    """
    Evaluate Ising energy directly from
    an asset-order binary bitstring.
    """
    x = bitstring_to_binary(
        bitstring
    )

    z = (
        1.0
        - 2.0 * x
    )

    energy = float(
        offset
    )

    for i, asset in enumerate(
        INVESTABLE_ASSETS
    ):
        h = float(
            linear.loc[
                linear["asset"]
                == asset,
                "h",
            ].iloc[0]
        )

        energy += (
            h
            * z[i]
        )

    for _, row in quadratic.iterrows():
        i = INVESTABLE_ASSETS.index(
            row["asset_i"]
        )

        j = INVESTABLE_ASSETS.index(
            row["asset_j"]
        )

        energy += (
            float(
                row["J"]
            )
            * z[i]
            * z[j]
        )

    return float(energy)


def solve_with_qaoa(
    model_name: str,
    eta: float,
    reps: int,
    linear: pd.DataFrame,
    quadratic: pd.DataFrame,
    offset: float,
    penalty: float,
    exact_solution: pd.Series,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    """
    Solve one Ising Hamiltonian with QAOA
    and compare the sampled distribution
    with the exact ground state.
    """
    algorithm_globals.random_seed = (
        RANDOM_SEED
    )

    sampler = StatevectorSampler(
        default_shots=SHOTS,
        seed=RANDOM_SEED,
    )

    optimizer = COBYLA(
        maxiter=MAX_ITERATIONS
    )

    hamiltonian = build_cost_hamiltonian(
        linear=linear,
        quadratic=quadratic,
        offset=offset,
    )

    initial_point = np.full(
        2 * reps,
        INITIAL_POINT_VALUE,
        dtype=float,
    )

    optimization_history = []

    def callback(
        evaluation_count,
        parameters,
        mean,
        metadata,
    ):
        optimization_history.append(
            {
                "evaluation_count":
                    evaluation_count,
                "mean_energy":
                    float(
                        mean
                    ),
            }
        )

    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps,
        initial_point=initial_point,
        callback=callback,
    )

    result = (
        qaoa.compute_minimum_eigenvalue(
            hamiltonian
        )
    )

    exact_bitstring = str(
        exact_solution[
            "bitstring"
        ]
    ).zfill(
        len(
            INVESTABLE_ASSETS
        )
    )

    exact_energy = float(
        exact_solution[
            "ising_energy"
        ]
    )

    sample_rows = []

    for state, probability in (
        result.eigenstate.items()
    ):
        bitstring = (
            qiskit_state_to_bitstring(
                state
            )
        )

        selected_assets = (
            selected_assets_from_bitstring(
                bitstring
            )
        )

        energy = calculate_ising_energy(
            bitstring=bitstring,
            linear=linear,
            quadratic=quadratic,
            offset=offset,
        )

        cardinality = len(
            selected_assets
        )

        sample_rows.append(
            {
                "model":
                    model_name,
                "eta":
                    eta,
                "qaoa_reps":
                    reps,
                "bitstring":
                    bitstring,
                "selected_assets":
                    ", ".join(
                        selected_assets
                    ),
                "cardinality":
                    cardinality,
                "feasible":
                    cardinality
                    == NUMBER_OF_SELECTED_ASSETS,
                "energy":
                    energy,
                "probability":
                    float(
                        probability
                    ),
                "exact_match":
                    bitstring
                    == exact_bitstring,
            }
        )

    samples = pd.DataFrame(
        sample_rows
    )

    samples = samples.sort_values(
        [
            "energy",
            "probability",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    best_sample = samples.iloc[0]

    probability_ranking = (
        samples.sort_values(
            [
                "probability",
                "energy",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    most_likely_sample = (
        probability_ranking.iloc[0]
    )

    exact_rows = samples[
        samples["bitstring"]
        == exact_bitstring
    ]

    if exact_rows.empty:
        ground_state_probability = 0.0
        ground_state_probability_rank = (
            np.nan
        )
    else:
        ground_state_probability = float(
            exact_rows.iloc[0][
                "probability"
            ]
        )

        exact_rank_rows = (
            probability_ranking[
                probability_ranking[
                    "bitstring"
                ]
                == exact_bitstring
            ]
        )

        ground_state_probability_rank = (
            int(
                exact_rank_rows.index[0]
            )
            + 1
        )

    feasible_probability = float(
        samples.loc[
            samples["feasible"],
            "probability",
        ].sum()
    )

    best_energy = float(
        best_sample[
            "energy"
        ]
    )

    optimality_gap = (
        best_energy
        - exact_energy
    )

    result_row = {
        "model":
            model_name,
        "eta":
            eta,
        "penalty":
            penalty,
        "qaoa_reps":
            reps,
        "shots":
            SHOTS,
        "max_iterations":
            MAX_ITERATIONS,
        "random_seed":
            RANDOM_SEED,
        "optimizer":
            "COBYLA",
        "initial_point_value":
            INITIAL_POINT_VALUE,
        "best_bitstring":
            best_sample[
                "bitstring"
            ],
        "best_selected_assets":
            best_sample[
                "selected_assets"
            ],
        "best_cardinality":
            int(
                best_sample[
                    "cardinality"
                ]
            ),
        "best_feasible":
            bool(
                best_sample[
                    "feasible"
                ]
            ),
        "best_energy":
            best_energy,
        "exact_bitstring":
            exact_bitstring,
        "exact_energy":
            exact_energy,
        "optimality_gap":
            optimality_gap,
        "exact_match":
            bool(
                best_sample[
                    "exact_match"
                ]
            ),
        "ground_state_probability":
            ground_state_probability,
        "ground_state_probability_rank":
            ground_state_probability_rank,
        "feasible_probability":
            feasible_probability,
        "most_likely_bitstring":
            most_likely_sample[
                "bitstring"
            ],
        "most_likely_selected_assets":
            most_likely_sample[
                "selected_assets"
            ],
        "most_likely_probability":
            float(
                most_likely_sample[
                    "probability"
                ]
            ),
        "most_likely_exact_match":
            bool(
                most_likely_sample[
                    "exact_match"
                ]
            ),
        "qaoa_expectation":
            float(
                np.real(
                    result.eigenvalue
                )
            ),
        "optimizer_evaluations":
            len(
                optimization_history
            ),
    }

    return (
        result_row,
        samples,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        standard_linear,
        standard_quadratic,
    ) = load_ising_coefficients(
        STANDARD_LINEAR_PATH,
        STANDARD_QUADRATIC_PATH,
    )

    (
        systemic_linear,
        systemic_quadratic,
    ) = load_ising_coefficients(
        SYSTEMIC_LINEAR_PATH,
        SYSTEMIC_QUADRATIC_PATH,
    )

    (
        standard_penalty,
        standard_qubo_constant,
        standard_offset,
    ) = load_ising_offset(
        CONSTANTS_PATH,
        "standard",
    )

    (
        systemic_penalty,
        systemic_qubo_constant,
        systemic_offset,
    ) = load_ising_offset(
        CONSTANTS_PATH,
        "systemic",
    )

    standard_exact = (
        load_exact_solution(
            VALIDATION_PATH,
            "standard",
        )
    )

    systemic_exact = (
        load_exact_solution(
            VALIDATION_PATH,
            "systemic",
        )
    )

    result_rows = []
    sample_frames = []

    for reps in QAOA_REPS_VALUES:
        (
            standard_result,
            standard_samples,
        ) = solve_with_qaoa(
            model_name="standard",
            eta=STANDARD_ETA,
            reps=reps,
            linear=standard_linear,
            quadratic=standard_quadratic,
            offset=standard_offset,
            penalty=standard_penalty,
            exact_solution=standard_exact,
        )

        (
            systemic_result,
            systemic_samples,
        ) = solve_with_qaoa(
            model_name="systemic",
            eta=SYSTEMIC_ETA,
            reps=reps,
            linear=systemic_linear,
            quadratic=systemic_quadratic,
            offset=systemic_offset,
            penalty=systemic_penalty,
            exact_solution=systemic_exact,
        )

        result_rows.extend(
            [
                standard_result,
                systemic_result,
            ]
        )

        sample_frames.extend(
            [
                standard_samples,
                systemic_samples,
            ]
        )

    qaoa_results = pd.DataFrame(
        result_rows
    )

    qaoa_samples = pd.concat(
        sample_frames,
        ignore_index=True,
    )

    qaoa_results.to_csv(
        OUTPUT_DIR
        / "qaoa_results.csv",
        index=False,
    )

    qaoa_samples.to_csv(
        OUTPUT_DIR
        / "qaoa_samples.csv",
        index=False,
    )

    print(
        "EPIC30 QAOA Solver"
    )
    print("-" * 72)

    print(
        "\nBackend: StatevectorSampler"
    )

    print(
        "Optimizer: COBYLA"
    )

    print(
        "QAOA repetitions:",
        ", ".join(
            f"p={reps}"
            for reps in QAOA_REPS_VALUES
        ),
    )

    print(
        f"Shots per run: {SHOTS}"
    )

    print(
        f"Maximum optimizer iterations: "
        f"{MAX_ITERATIONS}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Initial parameter value: "
        f"{INITIAL_POINT_VALUE}"
    )

    print(
        "\nAsset order:"
    )

    print(
        ", ".join(
            INVESTABLE_ASSETS
        )
    )

    for result in result_rows:
        print(
            "\n"
            + result["model"]
            .replace(
                "_",
                " ",
            )
            .title()
            + f" — p={result['qaoa_reps']}"
        )

        print("-" * 72)

        print(
            "Cardinality penalty:",
            f"{result['penalty']:.3f}",
        )

        print(
            "Best sampled assets:",
            result[
                "best_selected_assets"
            ],
        )

        print(
            "Best sampled bitstring:",
            result[
                "best_bitstring"
            ],
        )

        print(
            "Cardinality:",
            result[
                "best_cardinality"
            ],
        )

        print(
            "Feasible:",
            result[
                "best_feasible"
            ],
        )

        print(
            "Best sampled energy:",
            f"{result['best_energy']:.8f}",
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
            "Exact state sampled:",
            result[
                "exact_match"
            ],
        )

        print(
            "Ground-state probability:",
            f"{result['ground_state_probability']:.3%}",
        )

        print(
            "Ground-state probability rank:",
            result[
                "ground_state_probability_rank"
            ],
        )

        print(
            "Feasible probability:",
            f"{result['feasible_probability']:.3%}",
        )

        print(
            "Most likely bitstring:",
            result[
                "most_likely_bitstring"
            ],
        )

        print(
            "Most likely assets:",
            result[
                "most_likely_selected_assets"
            ],
        )

        print(
            "Most likely probability:",
            f"{result['most_likely_probability']:.3%}",
        )

        print(
            "Most likely is exact:",
            result[
                "most_likely_exact_match"
            ],
        )

        print(
            "QAOA expectation:",
            f"{result['qaoa_expectation']:.8f}",
        )

        print(
            "Optimizer evaluations:",
            result[
                "optimizer_evaluations"
            ],
        )

    print(
        "\nSaved results"
    )

    print("-" * 72)

    print(
        OUTPUT_DIR
        / "qaoa_results.csv"
    )

    print(
        OUTPUT_DIR
        / "qaoa_samples.csv"
    )


if __name__ == "__main__":
    main()