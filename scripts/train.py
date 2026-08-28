"""Entry point you *run*. Library code is imported from the installed package."""

import robotics
from helpers import greet
from robotics import quat_mul
from robotics.ik.differential import damped_lstsq

import numpy as np


def main() -> None:
    print(greet("lab"))
    print("robotics loaded from:", robotics.__file__)

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    print("quat_mul(identity, identity) =", quat_mul(identity, identity))

    J = np.eye(3)
    dx = np.array([0.1, 0.0, 0.0])
    print("damped_lstsq:", damped_lstsq(J, dx))


if __name__ == "__main__":
    main()
