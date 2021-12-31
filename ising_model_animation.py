from spin_lattice import SpinLattice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import sys

def main():
  if len(sys.argv) != 4:
        print("Usage python3 ising.animation.py N T Glauber/Kawasaki")
        sys.exit()
        
  # Parse the command line arguments required to animate the Ising model.
  lattice_dimensions = int(sys.argv[1])
  temperature = float(sys.argv[2])
  dynamics_type = sys.argv[3]
  
  # Instantiate an object of the SpinLattice class.
  ising_model = SpinLattice(lattice_dimensions, temperature)
  
  if dynamics_type == "Glauber":
    ising_model.animate_ising_model_glauber()
  elif dynamics_type == "Kawasaki":
    ising_model.animate_ising_model_kawasaki()
    
main()