import os
import sys
import time
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =====================================================================
# 1. PolyReLU Contestants
# =====================================================================

class CubicPolyReLU(nn.Module):
    """
    C1-smooth cubic PolyReLU.

    Parameters:
        a: Controls the slope at x = 0.
        b: Controls the negative-side linear slope.
    """

    def __init__(
        self,
        init_a: float = 0.50,
        init_b: float = 0.10,
        trainable_a: bool = True,
        trainable_b: bool = True,
    ):
        super().__init__()

        if trainable_a:
            self.a = nn.Parameter(
                torch.tensor(float(init_a))
            )
        else:
            self.register_buffer(
                "a",
                torch.tensor(float(init_a))
            )

        if trainable_b:
            self.b = nn.Parameter(
                torch.tensor(float(init_b))
            )
        else:
            self.register_buffer(
                "b",
                torch.tensor(float(init_b))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b_eff = torch.clamp(self.b, min=0.0)
        a_eff = self.a

        c1 = a_eff
        c2 = (1.0 - b_eff) / 4.0
        c3 = (
            1.0
            + b_eff
            - 2.0 * a_eff
        ) / 6.0

        c0 = (
            1.0
            - c3
            - c2
            - c1
        )

        x_clamped = torch.clamp(
            x,
            min=-1.0,
            max=1.0,
        )

        poly_region = (
            c3 * x_clamped ** 3
            + c2 * x_clamped ** 2
            + c1 * x_clamped
            + c0
        )

        p_minus_one = (
            -c3
            + c2
            - c1
            + c0
        )

        leaky_region = (
            b_eff * (x + 1.0)
            + p_minus_one
        )

        return torch.where(
            x > 1.0,
            x,
            torch.where(
                x >= -1.0,
                poly_region,
                leaky_region,
            ),
        )


class QuinticPolyReLU(nn.Module):
    """
    C2-smooth quintic PolyReLU.

    The polynomial smoothly joins:

        f(x) = b(x + 1) + f(-1), x < -1
        polynomial,                 -1 <= x <= 1
        f(x) = x,                    x > 1
    """

    def __init__(
        self,
        init_a: float = 0.50,
        init_b: float = 0.10,
        trainable_a: bool = True,
        trainable_b: bool = True,
    ):
        super().__init__()

        if trainable_a:
            self.a = nn.Parameter(
                torch.tensor(float(init_a))
            )
        else:
            self.register_buffer(
                "a",
                torch.tensor(float(init_a))
            )

        if trainable_b:
            self.b = nn.Parameter(
                torch.tensor(float(init_b))
            )
        else:
            self.register_buffer(
                "b",
                torch.tensor(float(init_b))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b_eff = torch.clamp(self.b, min=0.0)
        a_eff = self.a

        c1 = a_eff

        c2 = (
            1.0 - b_eff
        ) / 4.0

        c3 = (
            5.0
            + 3.0 * b_eff
            - 8.0 * a_eff
        ) / 8.0

        c5 = (
            3.0 * a_eff
            - 1.0
            - b_eff
        ) / 8.0

        c0 = (
            1.0
            - c5
            - c3
            - c2
            - c1
        )

        x_clamped = torch.clamp(
            x,
            min=-1.0,
            max=1.0,
        )

        x2 = x_clamped ** 2
        x4 = x2 ** 2

        poly_region = (
            c5 * x4 * x_clamped
            + c3 * x2 * x_clamped
            + c2 * x2
            + c1 * x_clamped
            + c0
        )

        p_minus_one = (
            -c5
            - c3
            + c2
            - c1
            + c0
        )

        leaky_region = (
            b_eff * (x + 1.0)
            + p_minus_one
        )

        return torch.where(
            x > 1.0,
            x,
            torch.where(
                x >= -1.0,
                poly_region,
                leaky_region,
            ),
        )


# =====================================================================
# 2. Architecture Zoo
# =====================================================================

class SimpleMLP(nn.Module):

    def __init__(self, activation_factory):
        super().__init__()

        self.net = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28 * 28, 256),
            activation_factory(),

            nn.Linear(256, 128),
            activation_factory(),

            nn.Linear(128, 64),
            activation_factory(),

            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):

    def __init__(self, activation_factory):
        super().__init__()

        self.net = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28 * 28, 512),
            activation_factory(),

            nn.Linear(512, 256),
            activation_factory(),

            nn.Linear(256, 128),
            activation_factory(),

            nn.Linear(128, 64),
            activation_factory(),

            nn.Linear(64, 32),
            activation_factory(),

            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):

    def __init__(self, activation_factory):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),

            activation_factory(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            activation_factory(),

            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 7 * 7,
                128,
            ),

            activation_factory(),

            nn.Linear(
                128,
                10,
            ),
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


class NanoGPTBlockMLP(nn.Module):
    """
    Gated FFN-style architecture.

    This is a PolyGLU-style block rather than literally NanoGPT.
    """

    def __init__(self, activation_factory):
        super().__init__()

        d_in = 28 * 28
        d_hid = 256

        self.flatten = nn.Flatten()

        self.w1 = nn.Linear(
            d_in,
            d_hid,
            bias=False,
        )

        self.w2 = nn.Linear(
            d_hid,
            10,
            bias=False,
        )

        self.w3 = nn.Linear(
            d_in,
            d_hid,
            bias=False,
        )

        self.activation = activation_factory()

    def forward(self, x):

        x = self.flatten(x)

        gate = self.activation(
            self.w1(x)
        )

        value = self.w3(x)

        out = self.w2(
            gate * value
        )

        return out


# =====================================================================
# 3. Activation Mathematics
# =====================================================================

def safe_name(name):
    """
    Convert a name into a filesystem-safe filename component.
    """

    result = str(name).lower()

    for char in [
        " ",
        "(",
        ")",
        ",",
        "=",
        ":",
        "/",
        "\\",
        "-",
        ".",
    ]:
        result = result.replace(
            char,
            "_",
        )

    while "__" in result:
        result = result.replace(
            "__",
            "_",
        )

    return result.strip("_")


def cubic_polyrelu_numpy(
    x,
    a,
    b,
):
    """
    NumPy implementation of CubicPolyReLU.
    """

    a = float(a)
    b = max(
        0.0,
        float(b),
    )

    c1 = a

    c2 = (
        1.0 - b
    ) / 4.0

    c3 = (
        1.0
        + b
        - 2.0 * a
    ) / 6.0

    c0 = (
        1.0
        - c3
        - c2
        - c1
    )

    x_clamped = np.clip(
        x,
        -1.0,
        1.0,
    )

    polynomial = (
        c3 * x_clamped ** 3
        + c2 * x_clamped ** 2
        + c1 * x_clamped
        + c0
    )

    p_minus_one = (
        -c3
        + c2
        - c1
        + c0
    )

    negative = (
        b * (x + 1.0)
        + p_minus_one
    )

    return np.where(
        x > 1.0,
        x,
        np.where(
            x >= -1.0,
            polynomial,
            negative,
        ),
    )


def quintic_polyrelu_numpy(
    x,
    a,
    b,
):
    """
    NumPy implementation of QuinticPolyReLU.
    """

    a = float(a)
    b = max(
        0.0,
        float(b),
    )

    c1 = a

    c2 = (
        1.0 - b
    ) / 4.0

    c3 = (
        5.0
        + 3.0 * b
        - 8.0 * a
    ) / 8.0

    c5 = (
        3.0 * a
        - 1.0
        - b
    ) / 8.0

    c0 = (
        1.0
        - c5
        - c3
        - c2
        - c1
    )

    x_clamped = np.clip(
        x,
        -1.0,
        1.0,
    )

    x2 = x_clamped ** 2
    x4 = x2 ** 2

    polynomial = (
        c5 * x4 * x_clamped
        + c3 * x2 * x_clamped
        + c2 * x2
        + c1 * x_clamped
        + c0
    )

    p_minus_one = (
        -c5
        - c3
        + c2
        - c1
        + c0
    )

    negative = (
        b * (x + 1.0)
        + p_minus_one
    )

    return np.where(
        x > 1.0,
        x,
        np.where(
            x >= -1.0,
            polynomial,
            negative,
        ),
    )


def activation_numpy(
    x,
    activation_name,
    a=None,
    b=None,
):
    """
    Evaluate an activation function using NumPy.
    """

    name = activation_name.lower()

    if "cubic polyrelu" in name:
        return cubic_polyrelu_numpy(
            x,
            a,
            b,
        )

    if "quintic polyrelu" in name:
        return quintic_polyrelu_numpy(
            x,
            a,
            b,
        )

    if name == "relu":
        return np.maximum(
            0.0,
            x,
        )

    if name == "leakyrelu":

        return np.where(
            x >= 0.0,
            x,
            0.01 * x,
        )

    if (
        "silu" in name
        or "swish" in name
    ):

        sigmoid = (
            1.0
            / (
                1.0
                + np.exp(-x)
            )
        )

        return x * sigmoid

    if name == "elu":

        return np.where(
            x >= 0.0,
            x,
            np.exp(x) - 1.0,
        )

    if name == "gelu":

        erf_values = np.array([
            math.erf(
                value / math.sqrt(2.0)
            )
            for value in x
        ])

        return (
            0.5
            * x
            * (
                1.0
                + erf_values
            )
        )

    raise ValueError(
        f"Unsupported activation: "
        f"{activation_name}"
    )


def derivative_numpy(
    x,
    activation_name,
    a=None,
    b=None,
):
    """
    Analytical derivative f'(x).
    """

    name = activation_name.lower()

    # -------------------------------------------------------------
    # ReLU
    # -------------------------------------------------------------

    if name == "relu":

        return np.where(
            x > 0.0,
            1.0,
            0.0,
        )

    # -------------------------------------------------------------
    # LeakyReLU
    # -------------------------------------------------------------

    if name == "leakyrelu":

        return np.where(
            x >= 0.0,
            1.0,
            0.01,
        )

    # -------------------------------------------------------------
    # SiLU / Swish
    # -------------------------------------------------------------

    if (
        "silu" in name
        or "swish" in name
    ):

        sigmoid = (
            1.0
            / (
                1.0
                + np.exp(-x)
            )
        )

        return (
            sigmoid
            * (
                1.0
                + x
                * (
                    1.0
                    - sigmoid
                )
            )
        )

    # -------------------------------------------------------------
    # ELU
    # -------------------------------------------------------------

    if name == "elu":

        return np.where(
            x >= 0.0,
            1.0,
            np.exp(x),
        )

    # -------------------------------------------------------------
    # GELU
    # -------------------------------------------------------------

    if name == "gelu":

        erf_values = np.array([
            math.erf(
                value / math.sqrt(2.0)
            )
            for value in x
        ])

        normal_pdf = (
            np.exp(
                -0.5 * x ** 2
            )
            / math.sqrt(
                2.0 * math.pi
            )
        )

        normal_cdf = (
            0.5
            * (
                1.0
                + erf_values
            )
        )

        return (
            normal_cdf
            + x * normal_pdf
        )

    # -------------------------------------------------------------
    # Cubic PolyReLU
    # -------------------------------------------------------------

    if "cubic polyrelu" in name:

        a = float(a)
        b = max(
            0.0,
            float(b),
        )

        c1 = a

        c2 = (
            1.0 - b
        ) / 4.0

        c3 = (
            1.0
            + b
            - 2.0 * a
        ) / 6.0

        inside = (
            3.0 * c3 * x ** 2
            + 2.0 * c2 * x
            + c1
        )

        return np.where(
            x > 1.0,
            1.0,
            np.where(
                x < -1.0,
                b,
                inside,
            ),
        )

    # -------------------------------------------------------------
    # Quintic PolyReLU
    # -------------------------------------------------------------

    if "quintic polyrelu" in name:

        a = float(a)
        b = max(
            0.0,
            float(b),
        )

        c1 = a

        c2 = (
            1.0 - b
        ) / 4.0

        c3 = (
            5.0
            + 3.0 * b
            - 8.0 * a
        ) / 8.0

        c5 = (
            3.0 * a
            - 1.0
            - b
        ) / 8.0

        inside = (
            5.0 * c5 * x ** 4
            + 3.0 * c3 * x ** 2
            + 2.0 * c2 * x
            + c1
        )

        return np.where(
            x > 1.0,
            1.0,
            np.where(
                x < -1.0,
                b,
                inside,
            ),
        )

    raise ValueError(
        f"Unsupported activation "
        f"derivative: {activation_name}"
    )


# =====================================================================
# 4. Plotting System
# =====================================================================

class ActivationPlotter:

    def __init__(
        self,
        output_dir="polyrelu_plots",
    ):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    # -----------------------------------------------------------------
    # Individual activation curve
    # -----------------------------------------------------------------

    def plot_activation_curve(
        self,
        activation_name,
        arena_name,
        a=None,
        b=None,
    ):

        x = np.linspace(
            -4.0,
            4.0,
            1200,
        )

        y = activation_numpy(
            x,
            activation_name,
            a,
            b,
        )

        plt.figure(
            figsize=(9, 5),
            dpi=300,
        )

        plt.plot(
            x,
            y,
            linewidth=2.0,
        )

        plt.axhline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            0.0,
            linewidth=0.8,
        )

        if "polyrelu" in activation_name.lower():

            plt.axvline(
                -1.0,
                linestyle=":",
                alpha=0.7,
            )

            plt.axvline(
                1.0,
                linestyle=":",
                alpha=0.7,
            )

        plt.title(
            f"{activation_name}\n"
            f"{arena_name}",
            fontsize=13,
            fontweight="bold",
        )

        plt.xlabel(
            "Input (x)",
        )

        plt.ylabel(
            "Activation f(x)",
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.tight_layout()

        filename = (
            f"{safe_name(arena_name)}_"
            f"{safe_name(activation_name)}_"
            f"curve.png"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved activation curve: {path}"
        )

    # -----------------------------------------------------------------
    # Individual derivative
    # -----------------------------------------------------------------

    def plot_activation_derivative(
        self,
        activation_name,
        arena_name,
        a=None,
        b=None,
    ):

        x = np.linspace(
            -4.0,
            4.0,
            1200,
        )

        y = derivative_numpy(
            x,
            activation_name,
            a,
            b,
        )

        plt.figure(
            figsize=(9, 5),
            dpi=300,
        )

        plt.plot(
            x,
            y,
            linewidth=2.0,
        )

        plt.axhline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            0.0,
            linewidth=0.8,
        )

        if "polyrelu" in activation_name.lower():

            plt.axvline(
                -1.0,
                linestyle=":",
                alpha=0.7,
            )

            plt.axvline(
                1.0,
                linestyle=":",
                alpha=0.7,
            )

        plt.title(
            f"{activation_name} Derivative\n"
            f"{arena_name}",
            fontsize=13,
            fontweight="bold",
        )

        plt.xlabel(
            "Input (x)",
        )

        plt.ylabel(
            "Derivative f'(x)",
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.tight_layout()

        filename = (
            f"{safe_name(arena_name)}_"
            f"{safe_name(activation_name)}_"
            f"derivative.png"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved derivative graph: {path}"
        )

    # -----------------------------------------------------------------
    # Combined activation comparison
    # -----------------------------------------------------------------

    def plot_activation_comparison(
        self,
        activations,
        arena_name,
    ):

        x = np.linspace(
            -4.0,
            4.0,
            1200,
        )

        plt.figure(
            figsize=(11, 6),
            dpi=300,
        )

        for activation in activations:

            name = activation["name"]

            y = activation_numpy(
                x,
                name,
                activation.get("a"),
                activation.get("b"),
            )

            plt.plot(
                x,
                y,
                linewidth=2.0,
                label=name,
            )

        plt.axhline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            0.0,
            linewidth=0.8,
        )

        plt.title(
            f"Activation Function Comparison\n"
            f"{arena_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xlabel(
            "Input (x)",
        )

        plt.ylabel(
            "Activation f(x)",
        )

        plt.xlim(
            -4.0,
            4.0,
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(
            loc="upper left",
        )

        plt.tight_layout()

        filename = (
            f"{safe_name(arena_name)}_"
            f"activation_comparison.png"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved activation comparison: {path}"
        )

    # -----------------------------------------------------------------
    # Combined derivative comparison
    # -----------------------------------------------------------------

    def plot_derivative_comparison(
        self,
        activations,
        arena_name,
    ):

        x = np.linspace(
            -4.0,
            4.0,
            1200,
        )

        plt.figure(
            figsize=(11, 6),
            dpi=300,
        )

        for activation in activations:

            name = activation["name"]

            y = derivative_numpy(
                x,
                name,
                activation.get("a"),
                activation.get("b"),
            )

            plt.plot(
                x,
                y,
                linewidth=2.0,
                label=f"{name} f'(x)",
            )

        plt.axhline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            0.0,
            linewidth=0.8,
        )

        plt.title(
            f"Activation Derivative Comparison\n"
            f"{arena_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xlabel(
            "Input (x)",
        )

        plt.ylabel(
            "Derivative f'(x)",
        )

        plt.xlim(
            -4.0,
            4.0,
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(
            loc="upper left",
        )

        plt.tight_layout()

        filename = (
            f"{safe_name(arena_name)}_"
            f"derivative_comparison.png"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved derivative comparison: {path}"
        )

    # -----------------------------------------------------------------
    # Learned PolyReLU curves across layers
    # -----------------------------------------------------------------

    def plot_learned_polyrelu_curves(
        self,
        layer_params,
        variant,
        arena_name,
    ):

        if not layer_params:
            return

        x = np.linspace(
            -4.0,
            4.0,
            1200,
        )

        plt.figure(
            figsize=(11, 6),
            dpi=300,
        )

        # ReLU reference
        plt.plot(
            x,
            np.maximum(
                0.0,
                x,
            ),
            linestyle="--",
            alpha=0.5,
            linewidth=1.5,
            label="Standard ReLU",
        )

        for layer_name, params in (
            layer_params.items()
        ):

            if variant == "cubic":

                y = cubic_polyrelu_numpy(
                    x,
                    params["a"],
                    params["b"],
                )

            else:

                y = quintic_polyrelu_numpy(
                    x,
                    params["a"],
                    params["b"],
                )

            plt.plot(
                x,
                y,
                linewidth=2.0,
                label=(
                    f"{layer_name} "
                    f"(a={params['a']:.4f}, "
                    f"b={params['b']:.4f})"
                ),
            )

        plt.axhline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            -1.0,
            linestyle=":",
            alpha=0.7,
        )

        plt.axvline(
            1.0,
            linestyle=":",
            alpha=0.7,
        )

        plt.title(
            f"Learned {variant.title()} PolyReLU "
            f"Across Layers\n{arena_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xlabel(
            "Input (x)",
        )

        plt.ylabel(
            "Activation f(x)",
        )

        plt.xlim(
            -4.0,
            4.0,
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(
            loc="upper left",
        )

        plt.tight_layout()

        filename = (
            f"{safe_name(arena_name)}_"
            f"{variant}_polyrelu_learned_curves.png"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved learned PolyReLU curves: {path}"
        )

    # -----------------------------------------------------------------
    # Learned PolyReLU derivatives across layers
    # -----------------------------------------------------------------

    def plot_learned_polyrelu_derivatives(
        self,
        layer_params,
        variant,
        arena_name,
    ):

        if not layer_params:
            return

        x = np.linspace(
            -4.0,
            4.0,
            1200,
        )

        plt.figure(
            figsize=(11, 6),
            dpi=300,
        )

        for layer_name, params in (
            layer_params.items()
        ):

            if variant == "cubic":
                activation_name = (
                    "Cubic PolyReLU"
                )
            else:
                activation_name = (
                    "Quintic PolyReLU"
                )

            y = derivative_numpy(
                x,
                activation_name,
                params["a"],
                params["b"],
            )

            plt.plot(
                x,
                y,
                linewidth=2.0,
                label=(
                    f"{layer_name} "
                    f"(a={params['a']:.4f}, "
                    f"b={params['b']:.4f})"
                ),
            )

        plt.axhline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            0.0,
            linewidth=0.8,
        )

        plt.axvline(
            -1.0,
            linestyle=":",
            alpha=0.7,
        )

        plt.axvline(
            1.0,
            linestyle=":",
            alpha=0.7,
        )

        plt.title(
            f"Learned {variant.title()} PolyReLU "
            f"Derivatives\n{arena_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.xlabel(
            "Input (x)",
        )

        plt.ylabel(
            "Derivative f'(x)",
        )

        plt.xlim(
            -4.0,
            4.0,
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend(
            loc="upper left",
        )

        plt.tight_layout()

        filename = (
            f"{safe_name(arena_name)}_"
            f"{variant}_polyrelu_learned_derivatives.png"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        plt.savefig(
            path,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"Saved learned PolyReLU derivatives: {path}"
        )


# =====================================================================
# 5. Benchmark Function
# =====================================================================

def evaluate_activation(
    model_factory,
    model_name,
    name,
    activation_factory,
    train_loader,
    test_loader,
    epochs=5,
    device="cpu",
):

    print(
        f"\n--- Testing Activation: {name} ---"
    )

    # -------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = model_factory(
        activation_factory
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    start_time = time.time()

    final_accuracy = 0.0
    final_loss = 0.0
    best_accuracy = 0.0

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # ---------------------------------------------------------
        # Evaluation
        # ---------------------------------------------------------

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                outputs = model(images)

                predicted = (
                    outputs.argmax(
                        dim=1
                    )
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        final_accuracy = (
            100.0
            * correct
            / total
        )

        avg_loss = (
            running_loss
            / len(train_loader)
        )

        final_loss = avg_loss

        best_accuracy = max(
            best_accuracy,
            final_accuracy,
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Test Accuracy: "
            f"{final_accuracy:.2f}%"
        )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Finished {name} in "
        f"{elapsed:.2f}s | "
        f"Final Accuracy: "
        f"{final_accuracy:.2f}%"
    )

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------

    plotter = ActivationPlotter()

    # -------------------------------------------------------------
    # Standard activation
    # -------------------------------------------------------------

    if "PolyReLU" not in name:

        plotter.plot_activation_curve(
            activation_name=name,
            arena_name=model_name,
        )

        plotter.plot_activation_derivative(
            activation_name=name,
            arena_name=model_name,
        )

    # -------------------------------------------------------------
    # PolyReLU
    # -------------------------------------------------------------

    layer_params = {}

    if "PolyReLU" in name:

        print(
            "\n--- Learned Parameters "
            "Across Layers ---"
        )

        layer_idx = 1

        for module in model.modules():

            if isinstance(
                module,
                (
                    CubicPolyReLU,
                    QuinticPolyReLU,
                ),
            ):

                final_a = (
                    module.a.item()
                )

                final_b = (
                    module.b.item()
                )

                clamped_b = max(
                    0.0,
                    final_b,
                )

                module_name = (
                    module.__class__.__name__
                )

                print(
                    f"Layer {layer_idx} "
                    f"({module_name}): "
                    f"learned a = "
                    f"{final_a:.6f} | "
                    f"learned b = "
                    f"{clamped_b:.6f}"
                )

                layer_params[
                    f"Layer {layer_idx}"
                ] = {
                    "a": final_a,
                    "b": clamped_b,
                }

                layer_idx += 1

        # ---------------------------------------------------------
        # Determine variant
        # ---------------------------------------------------------

        if "Cubic" in name:

            variant = "cubic"

            activation_label = (
                "Cubic PolyReLU"
            )

        else:

            variant = "quintic"

            activation_label = (
                "Quintic PolyReLU"
            )

        # ---------------------------------------------------------
        # Individual learned activation
        # ---------------------------------------------------------

        if layer_params:

            # Learned curves
            plotter.plot_learned_polyrelu_curves(
                layer_params=layer_params,
                variant=variant,
                arena_name=model_name,
            )

            # Learned derivatives
            plotter.plot_learned_polyrelu_derivatives(
                layer_params=layer_params,
                variant=variant,
                arena_name=model_name,
            )

            # Individual layer plots
            for layer_name, params in (
                layer_params.items()
            ):

                plotter.plot_activation_curve(
                    activation_name=(
                        activation_label
                    ),
                    arena_name=(
                        f"{model_name}_"
                        f"{layer_name}"
                    ),
                    a=params["a"],
                    b=params["b"],
                )

                plotter.plot_activation_derivative(
                    activation_name=(
                        activation_label
                    ),
                    arena_name=(
                        f"{model_name}_"
                        f"{layer_name}"
                    ),
                    a=params["a"],
                    b=params["b"],
                )

    return {
        "accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "loss": final_loss,
        "time": elapsed,
        "layer_params": layer_params,
    }


# =====================================================================
# 6. Main
# =====================================================================

def main():

    print(
        "=========================================="
    )

    print(
        "       PolyReLU Test Suite"
    )

    print(
        "=========================================="
    )

    # -------------------------------------------------------------
    # Architecture selection
    # -------------------------------------------------------------

    print(
        "\nSelect Architecture to Benchmark:"
    )

    print(
        "  1) Standard MLP "
        "(3 Hidden Layers)"
    )

    print(
        "  2) Deep MLP "
        "(5 Hidden Layers)"
    )

    print(
        "  3) Simple CNN "
        "(2 Conv2D + MaxPool Layers)"
    )

    print(
        "  4) Gated FFN "
        "(PolyGLU Style)"
    )

    model_choice = (
        input(
            "Enter choice (1-4, default 1): "
        ).strip()
        or "1"
    )

    model_map = {

        "1": (
            "SimpleMLP",
            SimpleMLP,
        ),

        "2": (
            "DeepMLP",
            DeepMLP,
        ),

        "3": (
            "SimpleCNN",
            SimpleCNN,
        ),

        "4": (
            "NanoGPTBlockMLP",
            NanoGPTBlockMLP,
        ),
    }

    model_name, selected_model_factory = (
        model_map.get(
            model_choice,
            (
                "SimpleMLP",
                SimpleMLP,
            ),
        )
    )

    # -------------------------------------------------------------
    # PolyReLU selection
    # -------------------------------------------------------------

    print(
        "\nSelect PolyReLU Variant / Mode:"
    )

    print(
        "  1) Cubic PolyReLU (C1 Smooth)"
    )

    print(
        "  2) Quintic PolyReLU (C2 Smooth)"
    )

    print(
        "  3) Duel Mode (Both Cubic & Quintic)"
    )

    poly_choice = (
        input(
            "Enter choice (1-3, default 3): "
        ).strip()
        or "3"
    )

    # -------------------------------------------------------------
    # Initial parameters
    # -------------------------------------------------------------

    try:

        val_a = float(
            input(
                "\nEnter initial value for "
                "'a' (default 0.50): "
            )
            or 0.50
        )

    except ValueError:

        print(
            "Invalid a. Using 0.50."
        )

        val_a = 0.50

    try:

        val_b = float(
            input(
                "Enter initial value for "
                "'b' (default 0.10): "
            )
            or 0.10
        )

    except ValueError:

        print(
            "Invalid b. Using 0.10."
        )

        val_b = 0.10

    train_a = (
        input(
            "Make parameter 'a' trainable "
            "via backprop? (y/n, default y): "
        )
        .strip()
        .lower()
        or "y"
    ) == "y"

    train_b = (
        input(
            "Make parameter 'b' trainable "
            "via backprop? (y/n, default y): "
        )
        .strip()
        .lower()
        or "y"
    ) == "y"

    # -------------------------------------------------------------
    # Baselines
    # -------------------------------------------------------------

    baseline_options = {

        "1": (
            "ReLU",
            lambda: nn.ReLU(),
        ),

        "2": (
            "GELU",
            lambda: nn.GELU(),
        ),

        "3": (
            "LeakyReLU",
            lambda: nn.LeakyReLU(
                negative_slope=0.01
            ),
        ),

        "4": (
            "SiLU (Swish)",
            lambda: nn.SiLU(),
        ),

        "5": (
            "ELU",
            lambda: nn.ELU(),
        ),
    }

    print(
        "\nSelect Baseline Activations "
        "to Benchmark Against:"
    )

    print("  1) ReLU")
    print("  2) GELU")
    print("  3) LeakyReLU")
    print("  4) SiLU (Swish)")
    print("  5) ELU")
    print("  6) Run ALL Baselines")
    print("  0) No Baselines (PolyReLU Only)")

    baseline_input = input(
        "\nEnter choice(s) separated by commas "
        "(e.g. '1, 2, 4' or '6' for all) "
        "[default: 4, 2]: "
    ).strip()

    selected_activations = {}

    if not baseline_input:

        selected_activations[
            "SiLU (Swish)"
        ] = baseline_options["4"][1]

        selected_activations[
            "GELU"
        ] = baseline_options["2"][1]

    elif (
        baseline_input == "6"
        or baseline_input.lower() == "all"
    ):

        for (
            name,
            factory,
        ) in baseline_options.values():

            selected_activations[
                name
            ] = factory

    elif (
        baseline_input == "0"
        or baseline_input.lower() == "none"
    ):

        pass

    else:

        choices = [
            c.strip()
            for c in baseline_input.split(",")
        ]

        for choice in choices:

            if choice in baseline_options:

                name, factory = (
                    baseline_options[choice]
                )

                selected_activations[
                    name
                ] = factory

    # -------------------------------------------------------------
    # Add PolyReLU contestants
    # -------------------------------------------------------------

    if poly_choice == "1":

        selected_activations[
            f"Cubic PolyReLU "
            f"(C1, a={val_a}, b={val_b})"
        ] = lambda: CubicPolyReLU(
            init_a=val_a,
            init_b=val_b,
            trainable_a=train_a,
            trainable_b=train_b,
        )

    elif poly_choice == "2":

        selected_activations[
            f"Quintic PolyReLU "
            f"(C2, a={val_a}, b={val_b})"
        ] = lambda: QuinticPolyReLU(
            init_a=val_a,
            init_b=val_b,
            trainable_a=train_a,
            trainable_b=train_b,
        )

    else:

        selected_activations[
            f"Cubic PolyReLU "
            f"(C1, a={val_a}, b={val_b})"
        ] = lambda: CubicPolyReLU(
            init_a=val_a,
            init_b=val_b,
            trainable_a=train_a,
            trainable_b=train_b,
        )

        selected_activations[
            f"Quintic PolyReLU "
            f"(C2, a={val_a}, b={val_b})"
        ] = lambda: QuinticPolyReLU(
            init_a=val_a,
            init_b=val_b,
            trainable_a=train_a,
            trainable_b=train_b,
        )

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    print(
        "\n--- Tournament Configuration ---"
    )

    print(
        f"Target Model        : "
        f"{model_name}"
    )

    print(
        f"PolyReLU Parameters : "
        f"Initial a = {val_a}, "
        f"Initial b = {val_b}"
    )

    print(
        f"Trainable Flags     : "
        f"Trainable 'a' = {train_a}, "
        f"Trainable 'b' = {train_b}"
    )

    print(
        f"Contestants         : "
        f"{', '.join(selected_activations.keys())}"
    )

    confirm = input(
        "\nDo you want to run the benchmark "
        "tournament? (y/n): "
    ).strip().lower()

    if confirm != "y":

        print(
            "Run cancelled by user. "
            "Exiting script."
        )

        sys.exit(0)

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    print(
        "\nStarting benchmark tournament...\n"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Running benchmarks on: "
        f"{device}"
    )

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------
    # Tournament
    # -------------------------------------------------------------

    results = {}

    for (
        name,
        activation_factory,
    ) in selected_activations.items():

        result = evaluate_activation(

            selected_model_factory,

            model_name,

            name,

            activation_factory,

            train_loader,

            test_loader,

            epochs=5,

            device=device,
        )

        results[name] = result

    # -------------------------------------------------------------
    # Build final activation comparison graphs
    #
    # For PolyReLU, use the LEARNED parameters from the run.
    # -------------------------------------------------------------

    comparison_activations = []

    for (
        name,
        result,
    ) in results.items():

        if (
            "Cubic PolyReLU"
            in name
        ):

            layer_params = (
                result["layer_params"]
            )

            if layer_params:

                # Use Layer 1 as the representative
                # curve for the global comparison.
                params = layer_params[
                    "Layer 1"
                ]

                comparison_activations.append({
                    "name": "Cubic PolyReLU",
                    "a": params["a"],
                    "b": params["b"],
                })

            else:

                comparison_activations.append({
                    "name": "Cubic PolyReLU",
                    "a": val_a,
                    "b": val_b,
                })

        elif (
            "Quintic PolyReLU"
            in name
        ):

            layer_params = (
                result["layer_params"]
            )

            if layer_params:

                params = layer_params[
                    "Layer 1"
                ]

                comparison_activations.append({
                    "name": "Quintic PolyReLU",
                    "a": params["a"],
                    "b": params["b"],
                })

            else:

                comparison_activations.append({
                    "name": "Quintic PolyReLU",
                    "a": val_a,
                    "b": val_b,
                })

        else:

            comparison_activations.append({
                "name": name,
                "a": None,
                "b": None,
            })

    # -------------------------------------------------------------
    # Combined plots
    # -------------------------------------------------------------

    plotter = ActivationPlotter()

    if comparison_activations:

        plotter.plot_activation_comparison(
            comparison_activations,
            model_name,
        )

        plotter.plot_derivative_comparison(
            comparison_activations,
            model_name,
        )

    # -------------------------------------------------------------
    # Final Results
    # -------------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "             FINAL RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        f"{'Activation':<50}"
        f"{'Final Acc.':>12}"
        f"{'Best Acc.':>12}"
        f"{'Loss':>12}"
        f"{'Time':>12}"
    )

    print(
        "-" * 98
    )

    for (
        name,
        result,
    ) in results.items():

        print(
            f"{name:<50}"
            f"{result['accuracy']:>11.2f}%"
            f"{result['best_accuracy']:>11.2f}%"
            f"{result['loss']:>12.4f}"
            f"{result['time']:>11.2f}s"
        )

    print(
        "=========================================="
    )

    print(
        "\nGraphs saved to:"
    )

    print(
        os.path.abspath(
            "polyrelu_plots"
        )
    )

    print(
        "\nBenchmark complete."
    )


# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    main()