import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib as mpl
import numpy as np

from .mesh_utils import MeshData


def create_wave_animation(
    mesh: MeshData, data: np.ndarray, fps: int
) -> ani.FuncAnimation:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        ax.set_zlim([-2, 2])
        ax.plot_trisurf(
            mesh.xs,
            mesh.ys,
            data[i, :],
            triangles=mesh.elements,
            cmap=mpl.colormaps["YlGnBu_r"],
        )

    return ani.FuncAnimation(fig, update, frames=len(data), interval=1000 / fps)  # type: ignore


class Visualizer1D:
    def __init__(self, xs: np.ndarray):
        self.xs = xs
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_xlim(0, 1)
        (self.line,) = self.ax.plot([], [], lw=2)

    def update(self, frame_data):
        self.line.set_data(self.xs, frame_data)
        return (self.line,)


class Visualizer2D:
    def __init__(self, mesh: MeshData):
        self.mesh = mesh
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")

    def update(self, frame_data):
        self.ax.clear()
        self.ax.set_zlim(-2, 2)
        return self.ax.plot_trisurf(
            self.mesh.xs,
            self.mesh.ys,
            frame_data,
            triangles=self.mesh.elements,
            cmap="viridis",
        )
