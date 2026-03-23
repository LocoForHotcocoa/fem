import numpy as np
import meshpy.triangle as triangle
from dataclasses import dataclass


@dataclass
class MeshData:
    vertices: np.ndarray
    elements: np.ndarray
    boundary_markers: np.ndarray

    @property
    def xs(self):
        return self.vertices[:, 0]

    @property
    def ys(self):
        return self.vertices[:, 1]


# mesh creation
def create_mesh(num_triangles: int) -> MeshData:
    """
    creates triangle mesh with meshpy.triangle library. Tries to estimate
    the number of boundary points using some random function that chatgpt gave me:

    `numBoundaryPoints = int(2 * np.sqrt(num_triangles))`

    TODO: figure out a better way to estimate `numBoundaryPoints`, `max_volume`, and `min_angle`.
            - could also just use a mesh_size or something that is more helpful in generation
    also TODO: figure out if there is a better mesh library for this
    """

    def round_trip_connect(start, end):
        result = [(i, i + 1) for i in range(start, end)]
        result.append((end, start))
        return result

    # Generate boundary points in a circular shape
    num_boundary_points = int(2 * np.sqrt(num_triangles))
    points = [
        (np.cos(angle), np.sin(angle))
        for angle in np.linspace(0, 2 * np.pi, num_boundary_points, endpoint=False)
    ]

    # Define mesh info
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points) - 1))

    # Estimate max_volume based on desired triangle count
    area_estimate = np.pi  # Approximate area of the circular domain
    max_volume = area_estimate / num_triangles  # Average triangle area

    # Build the mesh
    mesh = triangle.build(info, max_volume=max_volume, min_angle=30)

    # print(f"Generated {len(mesh.elements)} triangles (target: {num_triangles})")

    return MeshData(
        vertices=np.array(mesh.points),
        elements=np.array(mesh.elements),
        boundary_markers=np.array(mesh.point_markers, dtype=bool),
    )
