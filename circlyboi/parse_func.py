import numexpr as ne  # type: ignore
import numpy as np
from typing import Callable

"""
This is the best I could get from the internet. There were other options but they seemed unsafe.
will add more to safe_locals here when needed.
"""

safe_locals = {
    "e": np.e,
    "pi": np.pi,
}


def parse_circle_func(raw_func: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Parse 2D function from user using numexpr and numpy.

    To optimize for my usecase (triangles), input shape == output shape (i.e. `x[1], y[1] -> u[1]`).
    """
    try:

        def func(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return ne.evaluate(
                raw_func,
                local_dict={**safe_locals, "x": x, "y": y},
                out=np.empty_like(x),
            )

        dummy_x = np.array([0.0])
        dummy_y = np.array([0.0])
        func(dummy_x, dummy_y)

        return func
    except (NameError, TypeError, ValueError, ZeroDivisionError) as e:
        print(f"function syntax error or invalid variable: {e}")
        print("function is not valid, please try again.")
        raise e


def parse_line_func(raw_func: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Parse 1D function from user using numexpr and numpy.
    """
    try:

        def func(x: np.ndarray) -> np.ndarray:
            return ne.evaluate(
                raw_func, local_dict={**safe_locals, "x": x}, out=np.empty_like(x)
            )

        dummy_x = np.array([0.0])
        func(dummy_x)

        return func
    except (NameError, TypeError, ValueError, ZeroDivisionError) as e:
        print(f"function syntax error or invalid variable: {e}")
        print("function is not valid, please try again.")
        raise e


if __name__ == "__main__":
    x_dummy = np.array([2.0])
    y_dummy = np.array([1.0])

    while True:
        raw_func = input("please enter a function of x and y:\n")
        circle_func = parse_circle_func(raw_func)
        if circle_func is not None:
            print(circle_func(x_dummy, y_dummy))

        raw_func = input("please enter a function of x:\n")
        line_func = parse_line_func(raw_func)
        if line_func is not None:
            print(line_func(x_dummy))
