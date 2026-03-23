import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from typing import Callable
import logging
import pathlib

from .mesh_utils import create_mesh
from .problems import Problem1D, Problem2D
from .solver import WaveEngine
from .plotting import Visualizer1D, Visualizer2D

logger = logging.Logger(__name__)


def fem_circle(
    func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    num_triangles: int,
    iterations: int,
    c: float,
    dt: float,
    dir: pathlib.Path = pathlib.Path("animations"),
    show: bool = True,
):
    FPS_TARGET = 30
    filename = f"FEM_tri_{num_triangles}_i_{iterations}_dt_{dt}_c_{c}.mp4"

    # Calculate how many physics steps happen per animation frame
    step_size = max(1, int(1.0 / (dt * FPS_TARGET)))

    # 2. Setup Geometry & Math
    logger.debug("Generating Mesh...")
    mesh = create_mesh(num_triangles)

    logger.debug("Assembling FEM Matrices...")
    problem = Problem2D(mesh=mesh, c=c)
    M = problem.assemble_matrices()

    # 3. Initialize the Engine
    engine = WaveEngine(M=M, dt=dt, boundary_mask=problem.mask)

    # 4. Define Initial Conditions
    # Apply the specified function to the mesh's x and y coordinates
    u_start = func(mesh.xs, mesh.ys)

    # Ensure the boundary starts exactly at 0
    u_start[problem.mask] = 0.0

    # 5. Run Simulation
    logger.debug(f"Running {iterations} iterations...")
    data = engine.run_simulation(
        u_start=u_start, iterations=iterations, step_size=step_size
    )

    # 6. Visualize
    logger.debug("Rendering Animation...")
    vis = Visualizer2D(mesh=mesh)

    def animate_frame(i):
        return vis.update(data[i])

    anim = ani.FuncAnimation(
        vis.fig,
        animate_frame,  # type: ignore
        frames=len(data),
        interval=(1.0 / FPS_TARGET) * 1000,
        blit=False,
    )
    if show:
        plt.show()

    else:
        logger.debug(f"Saving animation to {dir / filename}...")
        dir.mkdir(exist_ok=True)
        # calculate the actual fps based on step size
        actual_fps = int(1.0 / (dt * step_size))

        writer = ani.FFMpegWriter(bitrate=5000, fps=actual_fps)
        anim.save(dir / filename, writer=writer)


def fem_line(
    func: Callable[[np.ndarray], np.ndarray],
    num_elements: int,
    iterations: int,
    c: float,
    dt: float,
    dir: pathlib.Path = pathlib.Path("animations"),
    show: bool = True,
):
    FPS_TARGET = 30
    filename = f"FEM_1D_{num_elements}_i_{iterations}_dt_{dt}_c_{c}.mp4"

    # Calculate how many physics steps happen per animation frame
    step_size = max(1, int(1.0 / (dt * FPS_TARGET)))

    logger.debug("Assembling FEM Matrices...")
    problem = Problem1D(n=num_elements, c=c)
    M = problem.assemble_matrices()

    # 3. Initialize the Engine
    engine = WaveEngine(M=M, dt=dt, boundary_mask=problem.mask)

    # 4. Define Initial Conditions
    # Apply the specified function to the mesh's x and y coordinates
    u_start = func(problem.xs)

    # Ensure the boundary starts exactly at 0
    u_start[problem.mask] = 0.0

    # 5. Run Simulation
    logger.debug(f"Running {iterations} iterations...")
    data = engine.run_simulation(
        u_start=u_start, iterations=iterations, step_size=step_size
    )

    # 6. Visualize
    logger.debug("Rendering Animation...")
    vis = Visualizer1D(xs=problem.xs)

    def animate_frame(i):
        (line,) = vis.update(data[i])
        return (line,)

    anim = ani.FuncAnimation(
        vis.fig,
        animate_frame,  # type: ignore
        frames=len(data),
        interval=(1.0 / FPS_TARGET) * 1000,
        blit=True,
    )

    if show:
        plt.show()

    else:
        logger.debug(f"Saving animation to {dir / filename}...")
        dir.mkdir(exist_ok=True)
        # calculate the actual fps based on step size
        actual_fps = int(1.0 / (dt * step_size))

        writer = ani.FFMpegWriter(bitrate=5000, fps=actual_fps)
        anim.save(dir / filename, writer=writer)
