import matplotlib.pyplot as plt
import numpy as np

def backward_explicit(phi, cfl):
    return (1-cfl+cfl*np.cos(phi))**2 + (cfl*np.sin(phi))**2

def forward_explicit(phi, cfl):
    return (1+cfl-cfl*np.cos(phi))**2 + (cfl*np.sin(phi))**2

def central_explicit(phi, cfl):
    return 1 + (cfl*np.sin(phi))**2

def leap_frog(phi, cfl):
    return -2*cfl*np.sin(phi)*np.sqrt(1-(cfl*np.sin(phi))**2) + 1

def lax_wendroff(phi, cfl):
    return cfl**4 * (np.cos(phi) - 1)**2 + cfl**2 * (2*np.cos(phi) + np.sin(phi)**2 - 2) + 1

def lax(phi, cfl):
    return np.cos(phi)**2 + (cfl*np.sin(phi))**2

def central_implicit(phi, cfl):
    return 1 / (1+(cfl*np.sin(phi)**2))

def crank_nicolson(phi, cfl):
    return ((2-(cfl*np.sin(phi))**2)/(2*(1+(cfl*np.sin(phi))**2)))**2 + ((3*np.sin(phi))/(2*(1+(cfl*np.sin(phi))**2)))**2

schemes = [
    {
        "name": "Backward explicit",
        "func": backward_explicit
    },
    {
        "name": "Forward explicit",
        "func": forward_explicit
    },
    {
        "name": "Central explicit",
        "func": central_explicit
    },
    {
        "name": "Leap-frog",
        "func": leap_frog
    },
    {
        "name": "Lax-Wendroff",
        "func": lax_wendroff
    },
    {
        "name": "Lax",
        "func": lax
    },
    {
        "name": "Central implicit",
        "func": central_implicit
    },
    {
        "name": "Crank-Nicolson",
        "func": crank_nicolson
    }
]

if __name__ == "__main__":
    cfls_explicit = [0.25, 0.5, 0.8, 1.0]
    cfls_implicit = [0.25, 0.5, 0.85, 1.0, 10.0]
    markers = ["o", "s", "D", "^", "v"]
    phi = np.linspace(0, np.pi, 1000)
    gradient = np.tile(np.linspace(1,0,512), (512, 1)).T

    for scheme in schemes:
        plt.figure(figsize=(12,7), dpi=100)
        cfls = cfls_explicit
        markers = ["o", "s", "D", "^"]

        if scheme["name"] == "Central implicit" or scheme["name"] == "Crank-Nicolson":
            cfls = cfls_implicit
            markers = ["o", "s", "D", "^", "v"]

        for cfl, marker in zip(cfls, markers):
            plt.plot(phi, scheme["func"](phi, cfl), marker=marker, markevery=50, label=f"CFL={cfl}")

        ymin, ymax = plt.ylim()

        plt.plot(phi, np.ones(1000), label='Stability threshold', color='blue', linestyle='dotted')
        plt.imshow(gradient, cmap='Blues', interpolation='bicubic', aspect='auto',
            extent=[0, np.pi, 0.75, 1.0], alpha=0.2)
        plt.text(np.pi/2, 1.1, 'Unstable', horizontalalignment='center', verticalalignment='center', alpha=0.5, fontstyle='italic')
        
        plt.ylim(ymin, 1.1*ymax)

        plt.xlabel(r"$\phi$")
        plt.ylabel(r"$|G|^2$")
        #plt.title("Amplification factor magnitude in relation to the phase angle")
        plt.title(f"Scheme: {scheme['name']}")
        plt.legend()
        # plt.show()
        plt.savefig(f"stability_{scheme['name'].lower().replace(' ', '_')}.png")
        plt.close()
