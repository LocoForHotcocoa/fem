import numpy as np
import numpy.linalg as lin
import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.animation as ani


def animate_on_line(
    iterations: int, c: float, num_elements: int, dt: float, dir: str, show: bool, func
) -> None:
    # Amount of elements
    N = num_elements

    # Amount of nodes
    n = N + 1

    # Element size
    h = 1.0 / N

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
    step_size = math.ceil(1.0 / (dt * fps_target))
    fps = 1.0 / (dt * step_size)

    num_frames = math.floor(iterations / step_size)
    total_time = num_frames / fps
    print(f"total time is: {total_time:.2f} seconds")
    # filename to save animation
    filename = f"FEM_linear_{num_elements}_i_{iterations}_dt_{dt}_c_{c}.mp4"  # animation file name

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

    # Time coefficient matrix
    T = np.zeros((n, n))

    T[0, 0] = 1  # Left boundary
    T[N, N] = 1  # Right boundary

    for i in range(1, N):
        for j in range(0, N + 1):
            if i == j:
                T[i, j] = (2.0 / 3.0) * h
            if abs(i - j) == 1:
                T[i, j] = (1.0 / 6.0) * h

    # Space coefficient matrix
    S = np.zeros((n, n))

    for i in range(1, N):
        for j in range(0, N + 1):
            if i == j:
                S[i, j] = 2.0 / h
            if abs(i - j) == 1:
                S[i, j] = -(1.0 / h)

    # A single time step from U(t) -> U(t+dt) and U'(t) -> U'(t+dt)
    def iteration(v, vDer):
        vNew = v + dt * vDer
        q = -c * c * S @ v
        r = lin.solve(T, q)
        vDerNew = vDer + dt * r
        return (vNew, vDerNew)

    # The real solution
    # def realU(x, t):
    #     return np.cos(2*np.pi*t)*np.sin(2*np.pi*x)

    # The initial value of the finite element problem
    u = np.zeros((n, 1))
    uDer = np.zeros((n, 1))

    # use user defined function
    for i in range(0, n):
        x = i * h
        u[i] = func(x)

    # calculating all values
    data = []
    data.append(u)
    # a = 0
    for i in range(0, iterations):
        (uNew, uDerNew) = iteration(u, uDer)
        u = uNew
        uDer = uDerNew
        if i % step_size == 0:
            data.append(u)

    spacing = np.linspace(0.0, 1.0, n)  # for FEM solution

    ## animation ------------------------------------------------------------------------------- ##

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-1.7, 1.7))
    (line,) = ax.plot([], [], lw=2)

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(spacing, data[i])

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = ani.FuncAnimation(
        fig, animate, frames=num_frames, interval=(1.0 / fps) * 1000
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

        writer = ani.FFMpegWriter(bitrate=5000, fps=int(fps))
        anim.save(dir + "/" + filename, writer=writer)
        print(f"saving animation to {dir}/{filename}")
