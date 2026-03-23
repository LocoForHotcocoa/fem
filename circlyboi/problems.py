# much if the math is taken from the paper:
#     - https://studenttheses.uu.nl/bitstream/handle/20.500.12932/29861/thesis.pdf?sequence=2
# and much of the code in this section is taken from (or at the very least inspired by)
# this repository:
#     - https://github.com/jeverink/BachelorsThesis/
# The original MIT license applies to this code

# Copyright (c) 2018 jeverink

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import numpy.linalg as lin

from .mesh_utils import MeshData


class Problem1D:
    def __init__(self, n, c):
        self.n = n
        self.h = 1.0 / (n - 1)
        self.xs = np.linspace(0, 1, n)
        self.c = c
        # boundary mask: first and last node
        self.mask = np.zeros(n, dtype=bool)
        self.mask[0] = self.mask[-1] = True

    def assemble_matrices(self) -> np.ndarray:
        """Creates Time and Space matrices."""
        h = self.h
        # since T & S are diagonal matrices, I am constructing them using np.diag
        # Construct T matrix
        # Create arrays for the diagonal values
        main_T = np.full(self.n, (2.0 / 3.0) * h)
        off_T = np.full(self.n - 1, (1.0 / 6.0) * h)

        # Construct the matrix by summing diagonals
        # k=0 is main, k=1 is upper, k=-1 is lower
        T = np.diag(main_T, k=0) + np.diag(off_T, k=1) + np.diag(off_T, k=-1)

        # Construct S Matrix
        main_S = np.full(self.n, 2.0 / h)
        off_S = np.full(self.n - 1, -1.0 / h)

        S = np.diag(main_S, k=0) + np.diag(off_S, k=1) + np.diag(off_S, k=-1)

        # Apply Boundary Conditions
        T[0, 0] = 1.0
        T[-1, -1] = 1.0

        # --- Solving Time ---
        # Equation becomes: {u_tt} = -c^2 * [T_inv] * [S] * {u}
        # M = -c^2 * [T_inv] * [S]
        return -(self.c**2) * (lin.inv(T) @ S)


class Problem2D:
    def __init__(self, mesh: MeshData, c: float):
        self.mesh = mesh
        self.c = c
        self.mask = mesh.boundary_markers
        self.n = len(mesh.vertices)

    def assemble_matrices(self) -> np.ndarray:
        """
        computes the M Matrix (The "Evolution Matrix")
        Since c, T, and S are all CONSTANTS (the mesh doesn't change shape),
        we can bake them all into a single matrix M.
        """

        # our canonical element is the triangle that spans the points:
        # (0,0), (0,1), and (1,0)
        #
        # in this element, we use a set of
        # 3 canonical basis functions phi(x,y) where:
        # phi_1 = 1 - x - y
        # phi_2 = x
        # phi_3 = y
        #
        # then, over our canonical element:
        #
        # integral of (phi_i * phi_j * dA)
        # gives us A[i,j]:
        # fmt: off
        A = np.array(
            [[1/12 , -1/24, -1/24],
            [-1/24, 1/4  , 1/8  ],
            [-1/24, 1/8  , 1/12 ]]
        )

        # 	and integral of (np.dot(grad(phi_i), grad(phi_j)) * dA)
        #	gives us Ad[i,j]:
        Ad = np.array(
            [[1   , -1/2, -1/2],
            [-1/2, 1/2 , 0   ],
            [-1/2, 0   , 1/2 ]]
        )
        # fmt: on
        # to transform any element into the canonical element, we need this:
        # J = (x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1)
        # this is discussed further in the referenced paper

        # all we need to do is create (n,n) T & S matrices
        # (n = # of vertices in mesh), and iterate over all triangles.
        #
        # vertices that are shared with multiple triangles will have their T & S values summed.
        #
        # T[n,m] += J*A[i,j]
        # S[n,m] += J*Ad[i,j]

        # print("populating matrices...\n")

        # number of elements
        # N = len(triangles)

        # number of points
        n = len(self.mesh.vertices)

        # local basis integrals
        T, S = np.zeros((n, n)), np.zeros((n, n))

        for inds in self.mesh.elements:
            # (ind1, ind2, ind3) are the indices for each point on 1 triangle.
            # xs[ind] and ys[ind] will give x and y values for that index.
            xs, ys = self.mesh.xs[inds], self.mesh.ys[inds]
            # calculate J for this specific triangle
            # fmt: on
            J = ((xs[1] - xs[0]) * (ys[2] - ys[0])) - (
                (xs[2] - xs[0]) * (ys[1] - ys[0])
            )
            # fmt: off
            # now cycle through each (i,j) pair in A, Ad
            # to update T, S with the specific J for the current element
            local_T = J * A
            local_S = J * Ad

            grid = np.ix_(inds, inds)
            T[grid] += local_T
            S[grid] += local_S

        # boundary condition:
        # if point is on boundary (bs[i] = 1),
        # then set S[i:] = 0, T[i:] = 0, T[i,i] = 1
        for i in range(self.n):
            if self.mask[i]:
                T[i, :] = 0
                S[i, :] = 0
                T[i, i] = 1

        return -(self.c**2) * (lin.inv(T) @ S)
