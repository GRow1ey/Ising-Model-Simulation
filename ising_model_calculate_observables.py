from spin_lattice import SpinLattice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
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
  
  if dynamical_rule == "Glauber":
    ising_model.calculate_observables(dynamical_rule)
    ising_model.plot_mean_energy_against_temperature(dynamical_rule)
    
main()