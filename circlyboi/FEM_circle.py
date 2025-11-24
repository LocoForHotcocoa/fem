import numpy as np
import numpy.linalg as lin
import meshpy.triangle as triangle # type: ignore
import math
import pathlib
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# mesh creation
def create_mesh(num_triangles: int) -> triangle.MeshInfo:
    """
    creates triangle mesh with meshpy.triangle library. Tries to estimate
    the number of boundary points using some random function that chatgpt gave me:

    `numBoundaryPoints = int(2 * np.sqrt(num_triangles))`
    
    TODO: figure out a better way to estimate `numBoundaryPoints`, `max_volume`, and `min_angle`.
    """

    def round_trip_connect(start, end):
        result = [(i, i + 1) for i in range(start, end)]
        result.append((end, start))
        return result

    # Generate boundary points in a circular shape
    numBoundaryPoints = int(2 * np.sqrt(num_triangles))
    points = [(np.cos(angle), np.sin(angle)) for angle in np.linspace(0, 2*np.pi, numBoundaryPoints, endpoint=False)]

    # Define mesh info
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(round_trip_connect(0, len(points) - 1))

    # Estimate max_volume based on desired triangle count
    area_estimate = np.pi  # Approximate area of the circular domain
    max_volume = area_estimate / num_triangles  # Average triangle area

    # Build the mesh
    mesh = triangle.build(info, max_volume=max_volume, min_angle=30)

    print(f"Generated {len(mesh.elements)} triangles (target: {num_triangles})")

    return mesh

def animate_on_circle(
        iterations: int, 
        c: float, 
        num_triangles: int, 
        dt: float, 
        dir: str, 
        show: bool, 
        func: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None:
    """
    creates animation from initial function (func) and a whole bunch of other parameters

    calculating FPS and skipped frames:
    - dt (float): ∆t between iterations in FEM
    - step_size (int): how many iterations of FEM per frame in animation
    - fps (float): `1 / (dt * step_size)` - fps of actual plotted animation
    - iterations: num of iterations in FEM

    we want fps to be ~30 (idk just feels right).
    actual fps will most likely be slightly higher because dealing with integer step_size but thats ok.

    example:
    >>> dt = 0.01
    >>> step_size = math.ceil(1.0/(dt*fps_target)) = 1/(0.01*30) = 4
    >>> fps = 1/(0.01*3) = 25

    how long will video be? how many frames?

    >>> num_frames = math.floor(iterations / step_size) # get rid of the last frame to avoid out of bounds
    >>> total_time = num_frames / fps
    """

    fps_target = 30
    step_size = math.ceil(1.0/(dt*fps_target))
    fps = 1.0/(dt*step_size)

    num_frames = math.floor(iterations / step_size)
    total_time = num_frames / fps
    print(f'total time is: {total_time:.2f} seconds')

    # save = True # choose to save the animation as a file
    # dir = 'animations' # folder name to store animations
    filename = f'FEM_tri_{num_triangles}_i_{iterations}_dt_{dt}_c_{c}.mp4' # animation file name

    mesh = create_mesh(num_triangles)

    # Extract list of vertices and triangles from mesh
    vertices = np.array(mesh.points)
    triangles = np.array(mesh.elements)

    # make things easy, seperate (x,y) tuple into seperate list of xs and ys for each vertex
    xs = vertices[:,0]
    ys = vertices[:,1]

    # extract boundary point information. for each vertex, 0 = inside, 1 = on boundary
    bs = np.array(mesh.point_markers, dtype=bool)

    ## NOTICE ---------------------------------------------------------------------------------- ##
    
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

    ## Math ------------------------------------------------------------------------------------ ##

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
    #	integral of (phi_i * phi_j * dA)
    #	gives us A[i,j]:
    A = [[1/12 , -1/24, -1/24],
        [-1/24, 1/4  , 1/8  ],
        [-1/24, 1/8  , 1/12]]
    
    # 	and integral of (np.dot(grad(phi_i), grad(phi_j)) * dA)
    #	gives us Ad[i,j]:
    Ad = [[1   , -1/2, -1/2],
         [-1/2, 1/2 , 0   ],
         [-1/2, 0   , 1/2 ]]
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

    print('populating matrices...\n')
    
    # number of elements
    # N = len(triangles)

    # number of points
    n = len(vertices)

    # local basis integrals

    T = np.zeros((n, n))
    S = np.zeros((n, n))

    for [ind1, ind2, ind3] in triangles:

        # (ind1, ind2, ind3) are the indices for each point on 1 triangle.
        # xs[ind] and ys[ind] will give x and y values for that index.

        inds = [ind1, ind2, ind3]
        # calculate J for this specific triangle
        J = (xs[ind2]-xs[ind1])*(ys[ind3]-ys[ind1]) - (xs[ind3] - xs[ind1])*(ys[ind2] - ys[ind1])

        # now cycle through each (i,j) pair in A, Ad 
        # to update T, S with the specific J for the current element
        for i in range(0, 3):
            for j in range(0, 3):
                T[inds[i], inds[j]] += J*A[i][j]
                S[inds[i], inds[j]] += J*Ad[i][j]


    # boundary condition:
    # if point is on boundary (bs[i] = 1), 
    # then set S[i:] = 0, T[i:] = 0, T[i,i] = 1
    for i in range(0, n):
        if(bs[i]):
            for j in range(0, n):
                S[i,j] = 0
                if(i == j):
                    T[i, j] = 1
                else:
                    T[i, j] = 0

    # iterate through time using first order approximation
    def iteration(v, vDer):
        vNew = v + dt*vDer
        q = -c*c*S@v
        r = lin.solve(T, q)
        vDerNew = vDer + dt*r
        return vNew, vDerNew

    u = np.zeros(n)
    uDer = np.zeros(n)

    # filling in initial data for each element, 
    # making sure that u[boundary] == 0.

    # calculate all values at once using numpy
    u_temp = func(xs, ys)
    
    # apply conditional logic with boundary points
    u[bs] = 0.0

    # fill in the rest from our calculated u_temp
    u[~bs] = u_temp[~bs]

    # set uDer to 0
    uDer[:] = 0.0

    # iteration
    print('iterating FEM...\n')

    data = np.empty((num_frames, n))
    data[0, :] = u.copy()
    frame_idx = 1

    for i in range(1, iterations + 1):
        u, uDer = iteration(u, uDer)

        if i % step_size == 0:
            if frame_idx < num_frames:
                data[frame_idx, :] = u.copy()
                frame_idx += 1

    print('plotting solution...\n')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    ## animation ------------------------------------------------------------------------------- ##

    def animate(i):
        ax.clear()
        ax.set_zlim([-2,2]) # arbitrary, you can change this if you want
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.plot_trisurf(xs, ys, data[i, :], triangles=triangles, cmap=mpl.colormaps['YlGnBu_r'])

    anim = ani.FuncAnimation(
        fig, 
        animate, # type: ignore
        frames=num_frames, 
        interval=(1.0/fps)*1000
    )
    
    if show:
        plt.show()

    else:
        save_dir = pathlib.Path(dir)
        save_dir.mkdir(exist_ok=True)

        writer=ani.FFMpegWriter(bitrate=5000, fps=int(fps))
        anim.save(save_dir / filename, writer=writer)
        print(f'saving animation to {save_dir / filename}')

