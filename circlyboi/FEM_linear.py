import numpy as np
import numpy.linalg as lin
import math
import pathlib
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.animation as ani

def animate_on_line(
        iterations: int, 
        c: float, 
        num_elements: int, 
        dt: float, 
        dir: str, 
        show: bool, 
        func: Callable[[np.ndarray], np.ndarray]
    ) -> None:

    # Amount of elements
    N = num_elements

    # Amount of nodes
    n = N + 1

    # Element size
    h = 1.0/N

    xs = np.linspace(0, 1, n)

    """    
    calculating FPS and skipped frames:
    - dt (float): ∆t between iterations in FEM
    - step_size (int): how many iterations of FEM per frame in animation
    - fps (float): `1 / (dt * step_size)` - fps of actual plotted animation
    - iterations: num of iterations in FEM

    we want fps to be ~30 (idk just feels right).
    actual fps will most likely be slightly higher because dealing with integer step_size but thats ok.

    example:
    ```
    dt = 0.01
    step_size = math.floor(1.0/(dt*fps_target)) = 1/(0.01*30) => 3.33 ~= 3
    fps = 1/(0.01*3) = 33.3
    ```

    how long will video be? how many frames?
    num_frames = math.floor(iterations / step_size) -- get rid of the last frame to avoid out of bounds
    total_time = num_frames / fps
    """

    fps_target = 30
    step_size = math.ceil(1.0/(dt*fps_target))
    fps = 1.0/(dt*step_size)

    num_frames = math.floor(iterations / step_size)
    total_time = num_frames / fps
    print(f'total time is: {total_time:.2f} seconds')
    # filename to save animation
    filename = f'FEM_linear_{num_elements}_i_{iterations}_dt_{dt}_c_{c}.mp4' # animation file name

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

    print('populating matrices...\n')
    # --- Creating Time and Space matrices ---
    # since T & S are diagonal matrices, I am constructing them using np.diag
    # Construct T matrix
    # Create arrays for the diagonal values
    main_T = np.full(n, (2.0/3.0) * h)
    off_T  = np.full(n - 1, (1.0/6.0) * h)

    # Construct the matrix by summing diagonals
    # k=0 is main, k=1 is upper, k=-1 is lower
    T = np.diag(main_T, k=0) + np.diag(off_T, k=1) + np.diag(off_T, k=-1)

    # Construct S Matrix
    main_S = np.full(n, 2.0 / h)
    off_S  = np.full(n - 1, -1.0 / h)

    S = np.diag(main_S, k=0) + np.diag(off_S, k=1) + np.diag(off_S, k=-1)

    # Apply Boundary Conditions
    T[0, 0] = 1.0
    T[N, N] = 1.0


    # --- Solving Time ---
    # Equation becomes: {u_tt} = -c^2 * [T_inv] * [S] * {u}
    # M = -c^2 * [T_inv] * [S]

    T_inv = lin.inv(T)
    M = -c*c * (T_inv @ S)
    # A single time step from U(t) -> U(t+dt) and U'(t) -> U'(t+dt)
    # (Euler's method)
    def iteration(u, uDer):
        # 1. calculate acceleration: a = M * u
        acc = M @ u

        # 2. update position (u)
        uNew = u + dt * uDer

        # 3. update velocity (uDer)
        uDerNew = uDer + dt * acc

        return uNew, uDerNew

    # The initial value of the finite element problem
    u = np.zeros(n)
    uDer = np.zeros(n)

    u_temp = func(xs)
    u[:] = u_temp[:]

    # BC (0 on endpoints)
    u[0] = 0.0
    u[-1] = 0.0
        

    # calculating all values
    print('iterating FEM...\n')

    data = np.empty((num_frames, n))
    data[0, :] = u.copy()
    frame_idx = 1


    # iterate through time steps
    for i in range(1, iterations + 1):
        u, uDer = iteration(u, uDer)
        # set BC every time, in case of floating point errors
        u[0] = u[-1] = 0.0
        uDer[0] = uDer[-1] = 0.0
        
        if i % step_size == 0:
            if frame_idx < num_frames:
                data[frame_idx, :] = u.copy()
                frame_idx += 1

    ## animation ------------------------------------------------------------------------------- ##
    
    print('plotting solution...\n')
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-1.7, 1.7)) # ylim is arbritrary
    line, = ax.plot([], [], lw=2)

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(xs, data[i])

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = ani.FuncAnimation(
        fig, 
        animate, # type: ignore
        frames=num_frames, 
        interval=(1.0/fps)*1000
    )
    # print(iterations)
    # print(len(data))
    # print(a)
    # print(iterations / step_size)
    if show:
        plt.show()

    else:
        save_dir = pathlib.Path(dir)
        save_dir.mkdir(exist_ok=True)
        
        writer=ani.FFMpegWriter(bitrate=5000, fps=int(fps))
        anim.save(dir + '/' + filename, writer=writer)
        print(f'saving animation to {dir}/{filename}')

