from spin_lattice import SpinLattice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["text.usetex"] = True
import sys


def main():
    if len(sys.argv) != 3:
        print("Usage python3 ising.animation.py N Glauber/Kawasaki")
        sys.exit()

    # Parse the command line arguments required to animate the Ising model.
    lattice_dimensions = int(sys.argv[1])
    dynamical_rule = sys.argv[2]

    # Instantiate an object of the SpinLattice class.
    ising_model = SpinLattice(lattice_dimensions)

    while run_simulation:
        if dynamical_rule == "Glauber":
            prompt = "Would you like to calculate observables or plot observables? ("
            prompt += "calc/plot/exit): "
        elif dynamical_rule == "Kawasaki":
            prompt = "Would you like to calculate observables or plot observables? ("
            prompt += "calc/plot/exit): "
