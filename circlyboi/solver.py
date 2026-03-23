import numpy as np

class WaveEngine:
    def __init__(self, M: np.ndarray, dt: float, boundary_mask: np.ndarray):
        self.M = M
        self.dt = dt
        self.boundary_mask = boundary_mask
    
    def step(self, u: np.ndarray, u_der: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """N-dimensional step function using Euler's method."""
        acc = self.M @ u

        # Euler integration
        u_new = u + self.dt * u_der
        u_der_new = u_der + self.dt * acc

        u_new[self.boundary_mask] = 0
        u_der_new[self.boundary_mask] = 0

        return u_new, u_der_new
    
    def run_simulation(self, u_start: np.ndarray, iterations: int, step_size: int):
        """Run N-dimensional simulation for specified iterations and step size (doesn't return all the data)."""
        n = len(u_start)
        u, u_der = u_start.copy(), np.zeros(n)

        num_frames = iterations // step_size
        data = np.zeros((num_frames, n))

        for i in range(iterations):
            u, u_der = self.step(u, u_der)
            if i % step_size == 0 and (i // step_size) < num_frames:
                data[i // step_size] = u.copy()
        
        return data