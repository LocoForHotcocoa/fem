from .parse_func import parse_line_func, parse_circle_func
from .FEM_circle import animate_on_circle
from .FEM_linear import animate_on_line

import typer
from typing_extensions import Annotated

app = typer.Typer()

arb_circle_func = Annotated[
    str, typer.Argument(help="arbitrary function of x and y that fits in a r=1 circle")
]
arb_line_func = Annotated[str, typer.Argument(help="arbitrary function of x")]

# simulation arguments
elements_op = Annotated[
    int, typer.Option(help="give approximate number of triangles to use in FEM")
]
it_op = Annotated[int, typer.Option(help="# of iterations with dt time step")]
c_op = Annotated[float, typer.Option(help="speed of sound on membrane")]
dt_op = Annotated[float, typer.Option(help="time step in seconds between FEM frames")]


dir_op = Annotated[str, typer.Option(help="directory to store animations")]
render_op = Annotated[
    bool,
    typer.Option(
        "--show/--save",
        help="show matplotlib window while rendering, or save to animations dir location with ffmpeg (must be installed)",
    ),
]


# def animate_2D(iterations: int, c: float, numTriangles: int, dt: float, dir: str, show: bool, save: bool, func) -> None:
@app.command()
def circle(
    func: arb_circle_func = "e - exp(x**2 + y**2)",
    num_elements: elements_op = 200,
    iterations: it_op = 20000,
    speed: c_op = 1.5,
    dt: dt_op = 0.001,
    dir: dir_op = "animations",
    show: render_op = True,
):
    try:
        user_func = parse_circle_func(func)
    except (NameError, TypeError, ValueError, ZeroDivisionError) as e:
        raise typer.Exit(1) from e

    print("all working good")
    animate_on_circle(iterations, speed, num_elements, dt, dir, show, user_func)


@app.command()
def line(
    func: arb_line_func = "sin(2*pi*x)",
    num_elements: elements_op = 50,
    iterations: it_op = 10000,
    speed: c_op = 0.25,
    dt: dt_op = 0.001,
    dir: dir_op = "animations",
    show: render_op = True,
):
    try:
        user_func = parse_line_func(func)
    except (NameError, TypeError, ValueError, ZeroDivisionError) as e:
        raise typer.Exit(1) from e

    print("all working good")
    animate_on_line(iterations, speed, num_elements, dt, dir, show, user_func)
