from spin_lattice import SpinLattice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import sys

def main():
  if len(sys.argv) != 5:
        print("Usage python3 ising.animation.py N Glauber/Kawasaki nsweeps with/without_auto-correlation times")
        sys.exit()
        
  # Parse the command line arguments required to animate the Ising model.
  lattice_dimensions = int(sys.argv[1])
  dynamical_rule = sys.argv[2]
  nsweeps = int(sys.argv[3])
  with_auto_correlation_times_condition = sys.argv[4] == "True"
  
  # Instantiate an object of the SpinLattice class.
  ising_model = SpinLattice(lattice_dimensions, nsweeps=nsweeps, with_auto_correlation_times_condition=with_auto_correlation_times_condition)
  #if not with_auto_correlation_times_condition:
   # ising_model.calculate_auto_correlation_time()
  
  if dynamical_rule == "Glauber":
    ising_model.calculate_observables(dynamical_rule)
    ising_model.plot_mean_energy_against_temperature(dynamical_rule)
    ising_model.plot_absolute_magnetisation_against_temperature()
    ising_model.plot_scaled_specific_heat_against_temperature(dynamical_rule)
    ising_model.plot_susceptibility_against_temperature()
  
    
main()