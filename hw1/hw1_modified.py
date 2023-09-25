import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
from tqdm import tqdm # pip install tqdm
import sys 

# u_j^{n+1}=  u_j^{n} - \sigma (u_{j}^n - u_{j-1}^n)
class explicit_backward: # 0
    def __call__(self, u, cfl): u[2:-2] -= cfl * (u[2:-2] - u[1:-3])

# u_j^{n+1}=  u_j^{n} - \sigma (u_{j+1}^n - u_{j-1}^n)
class explicit_forward: # 1
    def __call__(self, u, cfl): u[2:-2] -= cfl * (u[3:-1] - u[1:-3])

# u_j^{n+1}=  u_j^{n} - 0.5 \sigma (u_{j+1}^n - u_{j-1}^n)
class explicit_centered: # 2
    def __call__(self, u, cfl): u[2:-2] -= 0.5 * cfl * (u[3:-1] - u[1:-3])

# u_j^{n+1}=  u_j^{n-1} - \sigma (u_{j+1}^n - u_{j-1}^n)
class leap_frog: # 3
    u_prev = None 
    def __call__(self, u, cfl): 
        if self.u_prev is None: 
            self.u_prev = u.copy()
            scheme = explicit_backward()
            scheme(u, cfl) # on first step we use explicit backward
        else:
            u_next = self.u_prev[2:-2] - cfl * (u[3:-1] - u[1:-3])
            self.u_prev = u.copy()
            u[2:-2] = u_next

# u_j^{n+1}=  u_j^{n} - 0.5 \sigma (u_{j+1}^n - u_{j-1}^n) + 0.5 * \sigma^2 (u_{j+1}^n - 2 u_j^{n} + u_{j-1}^n)
class lax_wendroff: # 4
    def __call__(self, u, cfl): u[2:-2] = u[2:-2] - 0.5 * cfl * (u[3:-1] - u[1:-3]) + 0.5 * cfl**2 * (u[3:-1] - 2 * u[2:-2] + u[1:-3])

# u_j^{n+1}=  0.5 (u_{j+1}^n + u_{j-1}^n) - 0.5 * \sigma (u_{j+1}^n - u_{j-1}^n)
class lax: # 5
    def __call__(self, u, cfl): u[2:-2] = 0.5 * (u[3:-1] + u[1:-3]) - 0.5 * cfl * (u[3:-1] - u[1:-3])

# u_{j}^{n+1} = u_{j}^{n} - 0.5 \sigma (u_{j+1}^{n} - u_{j-1}^{n})
class hybrid0: # 6
    def __call__(self, u, cfl):  u[2:-2] -= 0.5 * cfl * (u[3:-1] - u[1:-3])

# u_{j}^{n+1} + 0.5 \sigma (u_{j+1}^{n+1} - u_{j-1}^{n+1}) = u_{j}^{n}
class hybrid1: # 7
    lhs = None
    rhs = None
    def __call__(self, u, cfl):
        # alllocation first time running
        # and build lhs matrix only once
        if (self.lhs is None):
            self.lhs = np.eye(len(u)-4)
            self.rhs = np.zeros(len(u)-4)
            np.fill_diagonal(self.lhs[1:, :-1], -0.5 * cfl) # lower diag
            np.fill_diagonal(self.lhs[:-1, 1:], 0.5 * cfl) # upper diag
            # coupling for periodic boundary condition
            self.lhs[0, -1] = -0.5 * cfl
            self.lhs[-1, 0] = 0.5 * cfl
        
        self.rhs = u[2:-2].copy()
        # u = np.linalg.solve(self.lhs, u)
        u[2:-2] = np.linalg.solve(self.lhs, self.rhs)

class crank_nicolson: # 8
    lhs = None
    rhs = None
    def __call__(self, u, cfl):
        # alllocation first time running
        # and build lhs matrix only once
        if (self.lhs is None):
            self.lhs = np.eye(len(u)-4)
            self.rhs = np.zeros(len(u)-4)
            np.fill_diagonal(self.lhs[1:, :-1], -0.25 * cfl) # lower diag
            np.fill_diagonal(self.lhs[:-1, 1:], 0.25 * cfl) # upper diag
            self.lhs[0, -1] = -0.25 * cfl
            self.lhs[-1, 0] = 0.25 * cfl
        
        self.rhs = u[2:-2] - 0.25 * cfl * (u[3:-1] - u[1:-3])

        u[2:-2] = np.linalg.solve(self.lhs, self.rhs)

# bonus
class implicit_upwind: # 9
    lhs = None
    rhs = None
    def __call__(self, u, cfl):
        # alllocation first time running
        # and build lhs matrix only once
        if (self.lhs is None):
            self.lhs = np.eye(len(u)-4) * (1 + cfl)
            self.rhs = np.zeros(len(u)-4)
            np.fill_diagonal(self.lhs[1:, :-1], - cfl) # lower diag
            # np.fill_diagonal(self.lhs, 0.5 * cfl) # diag
            self.lhs[0, -1] = - cfl
        
        self.rhs = u[2:-2].copy()
        # u = np.linalg.solve(self.lhs, u)
        u[2:-2] = np.linalg.solve(self.lhs, self.rhs)

class custom_simon: # 10
    u_prev = None
    i = 0 # iteration counter
    def __call__(self, u, cfl):
        if self.u_prev is None: # initial allocation on first call
            self.u_prev = np.zeros((4, len(u))) # store n, n-1, n-2 and n-3

        self.u_prev[0] = u.copy()
        if (self.i == 0):
            scheme = explicit_backward()
            scheme(u, cfl)
        elif (self.i == 1 or self.i == 2):
            scheme = leap_frog()
            scheme.u_prev = self.u_prev[1]
            scheme(u, cfl)
        else:
            # - 2*cfl*(u[:-4] -4*u[1:-3] + 3*u[2:-2])
            u[2:-2] = (18 * self.u_prev[1][2:-2] - 6*self.u_prev[2][2:-2] + self.u_prev[3][2:-2] - 10*u[2:-2])/3 - 2*cfl*(u[3:-1] - u[1:-3])

        # rolling 
        self.u_prev[3] = self.u_prev[2]
        self.u_prev[2] = self.u_prev[1]
        self.u_prev[1] = self.u_prev[0]
        self.i += 1

schemes = [explicit_backward, explicit_forward, explicit_centered, leap_frog, lax_wendroff, lax, hybrid0, hybrid1, crank_nicolson, implicit_upwind, custom_simon]

if __name__ == "__main__":
    assert len(sys.argv) <= 2, "Too many arguments"
    scheme_nb = int(sys.argv[1]) if (len(sys.argv) == 2) else 0
    assert scheme_nb >= 0 and scheme_nb <= len(schemes), "Invalid scheme number"

    # scheme = schemes[4] # can manually specify here
    plt.figure(figsize=(10, 5))

    cfl_values = [0.25, 0.5, 0.75, 1]  # Example list of CFL values. Adjust as needed.

    for cfl in cfl_values:
        scheme = schemes[scheme_nb]()

        x_interval = [0, 5]
        nb_pts = 400
        c = 35
        t_final = 1.75 / c  # this makes sure that the wave is at x = 2.5 at the end of the simulation
        t = 0

        # Calculated discretization parameters
        dx = (x_interval[1] - x_interval[0]) / (nb_pts - 1)
        dt = cfl * dx / c

        # Mesh
        x = np.linspace(x_interval[0], x_interval[1], nb_pts)
        u = np.zeros(nb_pts + 4, dtype=np.float32)  # +4 for ghosts cells on both ends

        # Step wave function
        u[:int(0.5 / dx) + 3] = 0.0  # +1 for the int floor rounding and + 1 for the ghost cell
        u[int(0.5 / dx) + 3:int(1.0 / dx) + 3] = 1.0
        u[int(1.0 / dx) + 3:] = 0.0

        with tqdm(total=int(t_final / dt)) as pbar:
            while t < t_final:
                scheme(u, cfl)
                u[0] = u[-4]
                u[1] = u[-3]
                u[-2] = u[2]
                u[-1] = u[3]
                t += dt
                pbar.update()

        # Plot the final state for each CFL value
        plt.plot(x, u[2:-2], label=f'CFL={cfl}')

    plt.xlabel('Location [m]')
    plt.ylabel('Wave Speed [m/s]')
    plt.legend()
    plt.title('Hybride Explicit-Implicit θ = 1 (1.f.iii)')
    plt.show()