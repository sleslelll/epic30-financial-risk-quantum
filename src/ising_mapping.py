from itertools import product
from pathlib import Path

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
    / "ising"
)


INVESTABLE_ASSETS = [
    "ETH",
    "BTC",
    "NASDAQ",
    "SP500",
    "GOLD",
    "USD",
]


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


def load_qubo_metadata(
    path: Path,
    model: str,
) -> tuple[float, float]:
    """
    Load the penalty and constant associated
    with a QUBO model.
    """
    constants = pd.read_csv(path)

    row = constants[
        constants["model"] == model
    ]

    if row.empty:
        raise ValueError(
            f"No QUBO metadata found for model: {model}"
        )

    penalty = float(
        row.iloc[0]["penalty"]
    )

    constant = float(
        row.iloc[0]["constant"]
    )

    return (
        penalty,
        constant,
    )


def load_exact_solution(
    path: Path,
) -> pd.Series:
    """
    Load the exact QUBO ground state from
    exhaustive 64-state validation.
    """
    states = pd.read_csv(
        path,
        dtype={
            "bitstring": str,
        },
    )

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


def qubo_to_ising(
    matrix: pd.DataFrame,
    constant: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Convert a full symmetric QUBO

        E(x) = x.T @ Q @ x + constant

    into the Ising form

        E(z) = offset
             + sum_i h_i z_i
             + sum_{i<j} J_ij z_i z_j

    using

        x_i = (1 - z_i) / 2.
    """
    q = matrix.to_numpy(
        dtype=float,
    )

    number_of_assets = len(
        INVESTABLE_ASSETS
    )

    linear = np.zeros(
        number_of_assets,
        dtype=float,
    )

    quadratic = np.zeros(
        (
            number_of_assets,
            number_of_assets,
        ),
        dtype=float,
    )

    for i in range(
        number_of_assets
    ):
        linear[i] = (
            -0.5
            * np.sum(
                q[i, :]
            )
        )

    for i in range(
        number_of_assets
    ):
        for j in range(
            i + 1,
            number_of_assets,
        ):
            quadratic[
                i,
                j,
            ] = (
                0.5
                * q[i, j]
            )

            quadratic[
                j,
                i,
            ] = quadratic[
                i,
                j,
            ]

    diagonal_sum = float(
        np.trace(q)
    )

    off_diagonal_sum = float(
        np.sum(
            np.triu(
                q,
                k=1,
            )
        )
    )

    offset = float(
        constant
        + 0.5 * diagonal_sum
        + 0.5 * off_diagonal_sum
    )

    return (
        linear,
        quadratic,
        offset,
    )


def direct_qubo_energy(
    x: np.ndarray,
    matrix: pd.DataFrame,
    constant: float,
) -> float:
    """
    Evaluate the original full symmetric
    QUBO representation directly.
    """
    return float(
        x
        @ matrix.to_numpy()
        @ x
        + constant
    )


def direct_ising_energy(
    z: np.ndarray,
    linear: np.ndarray,
    quadratic: np.ndarray,
    offset: float,
) -> float:
    """
    Evaluate the Ising representation directly.
    """
    energy = float(
        offset
        + np.dot(
            linear,
            z,
        )
    )

    for i in range(
        len(INVESTABLE_ASSETS)
    ):
        for j in range(
            i + 1,
            len(INVESTABLE_ASSETS),
        ):
            energy += (
                quadratic[
                    i,
                    j,
                ]
                * z[i]
                * z[j]
            )

    return float(energy)


def binary_to_bitstring(
    x: np.ndarray,
) -> str:
    """
    Convert a binary state into a bitstring
    using the fixed asset ordering.
    """
    return "".join(
        str(
            int(value)
        )
        for value in x
    )


def selected_assets_from_binary(
    x: np.ndarray,
) -> list[str]:
    """
    Return selected assets from a binary state.
    """
    return [
        asset
        for asset, value in zip(
            INVESTABLE_ASSETS,
            x,
        )
        if int(value) == 1
    ]


def validate_ising_mapping(
    model_name: str,
    matrix: pd.DataFrame,
    constant: float,
    linear: np.ndarray,
    quadratic: np.ndarray,
    offset: float,
    exact_solution: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    """
    Validate QUBO-Ising energy equivalence
    across all 64 binary states.
    """
    validation_rows = []

    for state in product(
        [0, 1],
        repeat=len(
            INVESTABLE_ASSETS
        ),
    ):
        x = np.array(
            state,
            dtype=float,
        )

        z = (
            1.0
            - 2.0 * x
        )

        qubo_energy = direct_qubo_energy(
            x=x,
            matrix=matrix,
            constant=constant,
        )

        ising_energy = direct_ising_energy(
            z=z,
            linear=linear,
            quadratic=quadratic,
            offset=offset,
        )

        bitstring = binary_to_bitstring(
            x
        )

        selected_assets = (
            selected_assets_from_binary(
                x
            )
        )

        validation_rows.append(
            {
                "model":
                    model_name,
                "bitstring":
                    bitstring,
                "selected_assets":
                    ", ".join(
                        selected_assets
                    ),
                "cardinality":
                    int(
                        np.sum(x)
                    ),
                "qubo_energy":
                    qubo_energy,
                "ising_energy":
                    ising_energy,
                "energy_check_error":
                    abs(
                        qubo_energy
                        - ising_energy
                    ),
            }
        )

    validation = pd.DataFrame(
        validation_rows
    )

    validation = validation.sort_values(
        [
            "qubo_energy",
            "bitstring",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    best = validation.iloc[0]

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
            "qubo_energy"
        ]
    )

    ground_state_match = (
        best["bitstring"]
        == exact_bitstring
    )

    ground_state_energy_error = abs(
        float(
            best["qubo_energy"]
        )
        - exact_energy
    )

    max_energy_error = float(
        validation[
            "energy_check_error"
        ].max()
    )

    result = {
        "model":
            model_name,
        "ground_state_bitstring":
            best[
                "bitstring"
            ],
        "ground_state_assets":
            best[
                "selected_assets"
            ],
        "ground_state_energy":
            float(
                best[
                    "qubo_energy"
                ]
            ),
        "exact_bitstring":
            exact_bitstring,
        "exact_energy":
            exact_energy,
        "ground_state_match":
            ground_state_match,
        "ground_state_energy_error":
            ground_state_energy_error,
        "max_energy_error":
            max_energy_error,
    }

    return (
        result,
        validation,
    )


def save_ising_coefficients(
    model_name: str,
    linear: np.ndarray,
    quadratic: np.ndarray,
) -> None:
    """
    Save Ising linear and quadratic coefficients.
    """
    linear_output = pd.DataFrame(
        {
            "asset":
                INVESTABLE_ASSETS,
            "h":
                linear,
        }
    )

    quadratic_rows = []

    for i in range(
        len(INVESTABLE_ASSETS)
    ):
        for j in range(
            i + 1,
            len(INVESTABLE_ASSETS),
        ):
            quadratic_rows.append(
                {
                    "asset_i":
                        INVESTABLE_ASSETS[i],
                    "asset_j":
                        INVESTABLE_ASSETS[j],
                    "J":
                        float(
                            quadratic[
                                i,
                                j,
                            ]
                        ),
                }
            )

    quadratic_output = pd.DataFrame(
        quadratic_rows
    )

    linear_output.to_csv(
        OUTPUT_DIR
        / f"{model_name}_ising_linear.csv",
        index=False,
    )

    quadratic_output.to_csv(
        OUTPUT_DIR
        / f"{model_name}_ising_quadratic.csv",
        index=False,
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

    standard_penalty, standard_constant = (
        load_qubo_metadata(
            CONSTANTS_PATH,
            "standard",
        )
    )

    systemic_penalty, systemic_constant = (
        load_qubo_metadata(
            CONSTANTS_PATH,
            "systemic_risk_aware",
        )
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

    (
        standard_linear,
        standard_quadratic,
        standard_offset,
    ) = qubo_to_ising(
        matrix=standard_matrix,
        constant=standard_constant,
    )

    (
        systemic_linear,
        systemic_quadratic,
        systemic_offset,
    ) = qubo_to_ising(
        matrix=systemic_matrix,
        constant=systemic_constant,
    )

    (
        standard_result,
        standard_validation,
    ) = validate_ising_mapping(
        model_name="standard",
        matrix=standard_matrix,
        constant=standard_constant,
        linear=standard_linear,
        quadratic=standard_quadratic,
        offset=standard_offset,
        exact_solution=standard_exact,
    )

    (
        systemic_result,
        systemic_validation,
    ) = validate_ising_mapping(
        model_name="systemic",
        matrix=systemic_matrix,
        constant=systemic_constant,
        linear=systemic_linear,
        quadratic=systemic_quadratic,
        offset=systemic_offset,
        exact_solution=systemic_exact,
    )

    save_ising_coefficients(
        model_name="standard",
        linear=standard_linear,
        quadratic=standard_quadratic,
    )

    save_ising_coefficients(
        model_name="systemic",
        linear=systemic_linear,
        quadratic=systemic_quadratic,
    )

    energy_validation = pd.concat(
        [
            standard_validation,
            systemic_validation,
        ],
        ignore_index=True,
    )

    ising_constants = pd.DataFrame(
        [
            {
                "model":
                    "standard",
                "penalty":
                    standard_penalty,
                "qubo_constant":
                    standard_constant,
                "ising_offset":
                    standard_offset,
            },
            {
                "model":
                    "systemic",
                "penalty":
                    systemic_penalty,
                "qubo_constant":
                    systemic_constant,
                "ising_offset":
                    systemic_offset,
            },
        ]
    )

    energy_validation.to_csv(
        OUTPUT_DIR
        / "energy_validation.csv",
        index=False,
    )

    ising_constants.to_csv(
        OUTPUT_DIR
        / "ising_constants.csv",
        index=False,
    )

    overall_max_error = float(
        energy_validation[
            "energy_check_error"
        ].max()
    )

    print(
        "EPIC30 QUBO-to-Ising Mapping"
    )
    print("-" * 72)

    print(
        "\nBinary-spin convention:"
    )
    print(
        "x_i = (1 - z_i) / 2"
    )
    print(
        "z_i = 1 - 2 x_i"
    )

    print(
        "\nAsset order:"
    )
    print(
        ", ".join(
            INVESTABLE_ASSETS
        )
    )

    for result, penalty, constant, offset in [
        (
            standard_result,
            standard_penalty,
            standard_constant,
            standard_offset,
        ),
        (
            systemic_result,
            systemic_penalty,
            systemic_constant,
            systemic_offset,
        ),
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
            "Cardinality penalty:",
            f"{penalty:.3f}",
        )

        print(
            "QUBO constant:",
            f"{constant:.12f}",
        )

        print(
            "Ising offset:",
            f"{offset:.12f}",
        )

        print(
            "Ground-state bitstring:",
            result[
                "ground_state_bitstring"
            ],
        )

        print(
            "Ground-state assets:",
            result[
                "ground_state_assets"
            ],
        )

        print(
            "Ground-state energy:",
            f"{result['ground_state_energy']:.8f}",
        )

        print(
            "Exact match:",
            result[
                "ground_state_match"
            ],
        )

        print(
            "Maximum energy error:",
            f"{result['max_energy_error']:.3e}",
        )

    print(
        "\nOverall validation"
    )
    print("-" * 72)

    print(
        "States checked:",
        len(
            energy_validation
        ),
    )

    print(
        "Maximum QUBO-Ising error:",
        f"{overall_max_error:.3e}",
    )

    if overall_max_error > 1e-10:
        raise RuntimeError(
            "QUBO-Ising energy validation failed."
        )

    if not (
        standard_result[
            "ground_state_match"
        ]
        and systemic_result[
            "ground_state_match"
        ]
    ):
        raise RuntimeError(
            "Ising ground state does not match "
            "the exact QUBO ground state."
        )

    print(
        "Energy equivalence: PASS"
    )

    print(
        "Ground-state equivalence: PASS"
    )

    print(
        "\nSaved results"
    )
    print("-" * 72)

    print(
        OUTPUT_DIR
        / "standard_ising_linear.csv"
    )

    print(
        OUTPUT_DIR
        / "standard_ising_quadratic.csv"
    )

    print(
        OUTPUT_DIR
        / "systemic_ising_linear.csv"
    )

    print(
        OUTPUT_DIR
        / "systemic_ising_quadratic.csv"
    )

    print(
        OUTPUT_DIR
        / "ising_constants.csv"
    )

    print(
        OUTPUT_DIR
        / "energy_validation.csv"
    )


if __name__ == "__main__":
    main()