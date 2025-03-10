import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
import pandas as pd
from scipy.optimize import curve_fit

class SpinLattice():
  """A class to represent the lattice of spins in the Ising Model."""
  
  def __init__(self, lattice_dimensions, temperature=1.0, J=1.0, nsweeps=10000, with_auto_correlation_times_condition=False):
    self.lattice_dimensions = lattice_dimensions
    self.spin_lattice = np.ones((lattice_dimensions, lattice_dimensions), dtype=float)
    self.previous_spin_lattice = []
    self.temperature = temperature
    self.J = J
    self.nsweeps = nsweeps
    self.itrial = 0
    self.jtrial = 0
    self.itrial_1 = 0
    self.jtrial_1 = 0
    self.with_auto_correlation_times_condition = with_auto_correlation_times_condition
    self.auto_correlation_time = 0
  
  def get_lattice_dimensions(self):
    """Method to return the dimensions of the n x n spin lattice."""
    return self.lattice_dimensions

  def get_spin_lattice(self):
    """Method to return the n x n spin lattice."""
    return self.spin_lattice

  def set_spin_lattice(self, spin_lattice):
    """Method to update the n x n spin lattice."""
    self.spin_lattice = spin_lattice
    
  def get_previous_spin_lattice(self):
    """Method to return the previous spin lattice."""
    return self.previous_spin_lattice
  
  def set_previous_spin_lattice(self, previous_spin_lattice):
    """Method to update the previous spin lattice attribute."""
    self.previous_spin_lattice = previous_spin_lattice

  def get_temperature(self):
    """Method to return the temperature of the system."""
    return self.temperature

  def set_temperature(self, temperature):
    """Method to update the temperature attribute."""
    self.temperature = temperature
    
  def get_J(self):
    """
    Method to return J which is the interaction strength 
    between nearest neighbor spins <i,j>.
    """
    return self.J

  def get_nsweeps(self):
    """Method to return nsweeps"""
    return self.nsweeps

  def get_itrial(self):
    """Method to return itrial."""
    return self.itrial
  
  def set_itrial(self, itrial):
    """Method to update the itrial attribute."""
    self.itrial = itrial

  def get_jtrial(self):
    """Method to return itrial."""
    return self.jtrial
  
  def set_jtrial(self, jtrial):
    """Method to update the jtrial attribute."""
    self.jtrial = jtrial

  def get_itrial_1(self):
    """Method to return itrial."""
    return self.itrial_1
  
  def set_itrial_1(self, itrial_1):
    """Method to update the itrial_1 attribute."""
    self.itrial_1 = itrial_1

  def get_jtrial_1(self):
    """Method to return itrial."""
    return self.jtrial_1
  
  def set_jtrial_1(self, jtrial_1):
    """Method to update the jtrial_1 attribute."""
    self.jtrial_1 = jtrial_1
  
  def get_auto_correlation_time(self):
    """Method to return the current auto-correlation time."""
    return self.auto_correlation_time
  
  def set_auto_correlation_time(self, auto_correlation_time):
    """Method to set the current auto-correlation time."""
    self.auto_correlation_time = auto_correlation_time
    
  def get_with_auto_correlation_times_condition(self):
    """
    Return the boolean value denoting whether
    to calculate the auto-correlation times.
    """
    return self.with_auto_correlation_times_condition
      
  def initialise_spin_lattice(self):
    """
    Initialise the spin lattice for animation of the Ising model
    to allow accurate calculation of observables.
    Half of the spins are initialised as up and half as down.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()

    for row in range(lattice_dimensions):
            for column in range(lattice_dimensions):
                # Check if the sum of the current row and column of the
                # spin lattice is even.
                if(np.mod(row + column, 2) == 0):
                    # If the sum of the current row and column is even
                    # the spin is set to up.
                    spin_lattice[row, column] = 1
                    # If the sum of the current row and column is not even
                    # the spin is set to down.
                else:
                    spin_lattice[row, column] = -1
    
  def calculate_spin_configuration_energy(self, itrial, jtrial):
    """
    Calculate the energy of the current spin configuration.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()
    J = self.get_J()
    
    # Find the spin values of the nearest neighbours, considering
    # boundary conditions.
    above_spin = spin_lattice[np.mod(itrial - 1, lattice_dimensions), jtrial]
    below_spin = spin_lattice[np.mod(itrial + 1, lattice_dimensions), jtrial]
    left_spin = spin_lattice[itrial, np.mod(jtrial - 1, lattice_dimensions)]
    right_spin = spin_lattice[itrial, np.mod(jtrial + 1, lattice_dimensions)]
    
    # Calculate the sum of the spins of the nearest neighbours. 
    sum_nearest_neighbours_spins = left_spin + right_spin + above_spin + below_spin
    
    # Calculate the energy of the current spin configuration.
    return -J * spin_lattice[itrial, jtrial] * sum_nearest_neighbours_spins
  
  def calculate_total_energy(self):
    """Calculate the total energy of the lattice of spins."""
    lattice_dimensions = self.get_lattice_dimensions()
    total_energy = 0
    
    # Calculate the sum of the energy of each spin configuration.
    for row in range(lattice_dimensions):
      for column in range(lattice_dimensions):
        # Divide by two to correct for double counting spin configuration
        # energies.
        total_energy += self.calculate_spin_configuration_energy(row, column) / 2
        
    return total_energy
  
  def calculate_boltzmann_weight(self, energy_difference):
    """Calculate the Boltzmann weight."""
    temperature = self.get_temperature()
    
    return np.exp(-energy_difference / temperature)
  
  def calculate_magnetisation(self):
    """Calculate the magnetisation of the spin lattice."""
    latttice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()
    sum_of_spins = 0
    
    for row in range(latttice_dimensions):
      for column in range(latttice_dimensions):
        sum_of_spins += spin_lattice[row, column]
        
    return sum_of_spins
  
  def select_candidate_state_glauber(self):
    """
    Select a candidate state for Glauber
    dynamics.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    
    # Select a spin randomly by generating two random
    # floating point values between 0.0 and 1.0 and
    # multiply by the lattice dimensions to create 
    # a row and column value to represent the position 
    # of a spin.
    spin_position = np.random.random(2) * lattice_dimensions
    
    # Return the floor of the row and column floating
    # point values and round it to the nearest integer.
    itrial = int(np.floor(spin_position[0]))
    jtrial = int(np.floor(spin_position[1]))
    
    # Update the itrial and jtrial attributes with the
    # randomly selected spin position.
    self.set_itrial(itrial)
    self.set_jtrial(jtrial)
    
  def select_candidate_state_kawasaki(self):
    """
    Select two spins at random to be switched
    for Kawasaki dynamics.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()
    
    while True:
      # Select two spins at random by generating four
      # random floating point values between 0.0 and 1.0
      # and multiply by the lattice dimensions to create
      # two rows and two column values to represent the
      # positions of two spins in the lattice.
      spin_positions = np.random.random(4) * lattice_dimensions
      
      # Return the floor of the floating point values of 
      # the rows and columns and round them to the nearest
      # integer.
      itrial = int(np.floor(spin_positions[0]))
      jtrial = int(np.floor(spin_positions[1]))
      itrial_1 = int(np.floor(spin_positions[2]))
      jtrial_1 = int(np.floor(spin_positions[3]))
      
      if (spin_lattice[itrial, jtrial] != spin_lattice[itrial_1, jtrial_1]) and (itrial != itrial_1 and jtrial != jtrial_1):
        break
        
    
    # Update the itrial, jtrial, itrial_1 and jtrial_1
    # attributes with the randomly selected spin positions.
    self.set_itrial(itrial)
    self.set_jtrial(jtrial)
    self.set_itrial_1(itrial_1)
    self.set_jtrial_1(jtrial_1)
    
  def calculate_energy_difference_glauber(self):
    """
    Calculate the difference between the 
    original state and the candidate state.
    """
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    
    # Calculate the difference between the
    # candidate state and the current state.
    energy_current_state = self.calculate_spin_configuration_energy(itrial, jtrial)
    
    energy_candidate_state = -energy_current_state
    
    return energy_candidate_state - energy_current_state
  
  def nearest_neighbours_check_kawasaki(self):
    """
    Determine if the original state and candidate
    state are nearest neighbours.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    itrial_1 = self.get_itrial_1()
    jtrial_1 = self.get_jtrial_1()
    
    spin_1_position = [itrial, jtrial]
    spin_2_position = [itrial_1, jtrial_1]
    
    # Find nearest neighbours of first spin.
    left_neighbour = [np.mod(itrial - 1, lattice_dimensions), jtrial]
    right_neighbour = [np.mod(itrial + 1, lattice_dimensions), jtrial]
    above_neighbour = [itrial, np.mod(jtrial - 1, lattice_dimensions)]
    below_neighbour = [itrial, np.mod(jtrial + 1, lattice_dimensions)]
    
    first_spin_neighbours = [left_neighbour, right_neighbour, above_neighbour, below_neighbour]
    
    # Find nearest neighbours of second spin.
    above_neighbour = [np.mod(itrial_1 - 1, lattice_dimensions), jtrial_1]
    below_neighbour = [np.mod(itrial_1 + 1, lattice_dimensions), jtrial_1]
    left_neighbour = [itrial_1, np.mod(jtrial_1 - 1, lattice_dimensions)]
    right_neighbour = [itrial_1, np.mod(jtrial_1 + 1, lattice_dimensions)]
    
    second_spin_neighbours = [left_neighbour, right_neighbour, above_neighbour, below_neighbour]
    
    if (spin_1_position in second_spin_neighbours) or (spin_2_position in first_spin_neighbours):
      return True
    
  def calculate_energy_difference_kawasaki(self):
    """
    Calculate the energy difference from swapping 
    two spins in the lattice.
    """
    spin_lattice = self.get_spin_lattice()
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    itrial_1 = self.get_itrial_1()
    jtrial_1 = self.get_jtrial_1()
    
    # Determine whether the two spins are nearest neighbours
    # or have equal spin values.
    if self.nearest_neighbours_check_kawasaki():
      return 0
    else:
      energy_current_state = self.calculate_spin_configuration_energy(itrial, jtrial) + self.calculate_spin_configuration_energy(itrial_1, jtrial_1)
      
      energy_candidate_state = -self.calculate_spin_configuration_energy(itrial, jtrial) - self.calculate_spin_configuration_energy(itrial_1, jtrial_1)
      
      return energy_candidate_state - energy_current_state
  
  def metropolis_algorithm_glauber(self):
    """
    Determine whether a spin flip will be accepted,
    with probability equal to the minimum of 1 or
    the Boltzmann weight.
    """
    spin_lattice = self.get_spin_lattice()
    energy_difference = self.calculate_energy_difference_glauber()
    boltzmann_weight = self.calculate_boltzmann_weight(energy_difference)
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    
    # Generate a random number between 0 and 1.
    random_number = np.random.random()
    
    # If the energy difference is less than or equal
    # to 1 the spin flip is accepted.
    if energy_difference <= 0:
      spin_lattice[itrial, jtrial] *= -1
    # If the random number is less than or equal to
    # the Boltzmann weight then the spin flip is
    # also accepted.
    elif random_number <= boltzmann_weight:
      spin_lattice[itrial, jtrial] *= -1
      
  def metropolis_algorithm_kawasaki(self):
    """
    Determine whether a spin swap will be accepted
    according to the Metropolis algorithm, i.e.
    the minimum of 1 or the Boltzmann weight.
    """
    spin_lattice = self.get_spin_lattice()
    energy_difference = self.calculate_energy_difference_kawasaki()
    boltzmann_weight = self.calculate_boltzmann_weight(energy_difference)
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    itrial_1 = self.get_itrial_1()
    jtrial_1 = self.get_jtrial_1()
    
    # Generate a random number between 0 and 1.
    random_number = np.random.random()
    
    # If the energy difference is less than or equal
    # to 1 the spin swap is accepted.
    if energy_difference <= 0:
      spin_lattice[itrial, jtrial] *= -1
      spin_lattice[itrial_1, jtrial_1] *= -1
    # If the random number is less than or equal to
    # the Boltzmann weight then the spin swap is
    # also accepted
    elif random_number <= boltzmann_weight:
      spin_lattice[itrial, jtrial] *= -1
      spin_lattice[itrial_1, jtrial_1] *= -1
      
  def animate_ising_model_glauber(self):
    """
    Animate the Ising model using Glauber
    dynamics.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()
    nsweeps = self.get_nsweeps()
    
    # Initialise the spin lattice.
    self.initialise_spin_lattice_kawasaki()
    
    # Initialise the plot for the animation.
    figure = plt.figure()
    image = plt.imshow(spin_lattice, animated = True)
    
    # Run the animation.
    for sweep in range(nsweeps):
      for row in range(lattice_dimensions):
        for column in range(lattice_dimensions):
          self.select_candidate_state_glauber()
          energy_difference = self.calculate_energy_difference_glauber()
          self.metropolis_algorithm_glauber()
          
      # Write the positions and spin values of the lattice
      # to a data file every ten sweeps.
      if not np.mod(sweep, 10) and sweep > 400:
        update_string = "Current sweep: {}, Energy Difference: {}".format(sweep, energy_difference)
        print(update_string)
        
        with open("spins_glauber.dat", "a") as file_object:
          for row in range(lattice_dimensions):
            for column in range(lattice_dimensions):
              file_object.write("%d %d %lf\n" % (row, column, spin_lattice[row, column]))

        # Clear the current axes.
        plt.cla()
        
        # Update the animation.
        image = plt.imshow(spin_lattice, animated = True)
        plt.draw()
        plt.pause(0.001)
        
  def animate_ising_model_kawasaki(self):
    """
    Animate the Ising model using Kawasaki
    dynamics.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    spin_lattice = self.get_spin_lattice()
    nsweeps = self.get_nsweeps()
    
    # Initialise the spin lattice.
    self.initialise_spin_lattice_kawasaki()
    
    # Initialise the plot for the animation.
    figure = plt.figure()
    image = plt.imshow(spin_lattice, animated = True)
    
    # Run the animation.
    for sweep in range(nsweeps):
      for row in range(lattice_dimensions):
        for column in range(lattice_dimensions):
          self.select_candidate_state_kawasaki()
          energy_difference = self.calculate_energy_difference_kawasaki()
          self.metropolis_algorithm_kawasaki()
          
      # Write the positions and spin values of the lattice
      # to a data file every ten sweeps.
      if not np.mod(sweep, 10):
        update_string = "Current sweep: {}, Energy Difference: {}".format(sweep, energy_difference)
        print(update_string)
        
        with open("spins_glauber.dat", "w") as file_object:
          for row in range(lattice_dimensions):
            for column in range(lattice_dimensions):
              file_object.write("%d %d %lf\n" % (row, column, spin_lattice[row, column]))

        # Clear the current axes.
        plt.cla()
        
        # Update the animation.
        image = plt.imshow(spin_lattice, animated = True)
        plt.draw()
        plt.pause(0.001)
    
  def calculate_error(self, observable_data):
    """
    A method to calculate the error on a set of data using the jackknife resampling.
    """
    length_observable_data = len(observable_data)
    xi_bar = []
    
    for i in range(length_observable_data):
      xi = 0
      
      for j in range(length_observable_data):
        if i != j:
          xi += observable_data[j]
          
      xi /= length_observable_data - 1
      xi_bar.append(xi)
      
    xi_bar = np.array(xi_bar)
    
    mean_xi_bar = np.mean(xi_bar)
    
    variance_xi_bar = np.sum(np.square(xi_bar - mean_xi_bar)) * (length_observable_data - 1) / length_observable_data
    
    return np.sqrt(variance_xi_bar) / np.sqrt(length_observable_data)
    
  def calculate_scaled_specific_heat_capacity(self, energies):
    """
    A method to calculate the scaled specific heat capacity.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    temperature = self.get_temperature()
    mean_squared_energy = np.mean(np.square(energies))
    mean_energy_squared = np.square(np.mean(energies))
    
    return (mean_squared_energy - mean_energy_squared) / ((lattice_dimensions**2) * (temperature ** 2))
    
  def calculate_susceptibility(self, magnetisations):
    """
    A method to calculate the susceptibility.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    temperature = self.get_temperature()
    
    mean_squared_magnetisations = np.mean(np.square(magnetisations))
    squared_mean_magnetisations = np.square(np.mean(magnetisations))
    
    return (mean_squared_magnetisations-squared_mean_magnetisations) / ((lattice_dimensions**2) * temperature)
    
    
  def calculate_observables_glauber_without_auto_correlation_times(self):
    """
    Calculate the observables of the Ising model without using the
    auto-correlation times.
    """
    nsweeps = self.get_nsweeps()
    lattice_dimensions = self.get_lattice_dimensions()
    temperature = self.get_temperature()
    spin_lattice = self.get_spin_lattice()
    total_energies = []
    magnetisations = []
    
    for sweep in range(nsweeps+101):
      for row in range(lattice_dimensions):
        for column in range(lattice_dimensions):
          self.select_candidate_state_glauber()
          self.calculate_energy_difference_glauber()
          self.metropolis_algorithm_glauber()
      
      # Write the positions and spin values of the lattice
      # to a data file every ten sweeps.
      if sweep > 100:
        with open("Spins_Data/Spins_Data_" + str(np.round(temperature, 2)) + "/spins_glauber_" + str(np.round(temperature, 2)) + "_sweeps=" + str(sweep) + ".dat", "a") as file_object:
          for row in range(lattice_dimensions):
            for column in range(lattice_dimensions):
              file_object.write("%d %d %lf\n" % (row, column, spin_lattice[row, column]))
        # Calculate the mean total energy and magnetisation 
        # of the current state.
        current_energy = self.calculate_total_energy()
        current_magnetisation = self.calculate_magnetisation()
          
        # Append the current mean total energy and magnetisation
        # to the lists of mean total energies and magnetisations.
        total_energies.append(current_energy)
        magnetisations.append(current_magnetisation)
      #elif not calculate_auto_correlation_times_decision:
    
    mean_total_energy = np.mean(np.array(total_energies))
    mean_total_energy_error = self.calculate_error(np.array(total_energies))
    mean_absolute_magnetisation = np.mean(np.abs(np.array(magnetisations)))
    mean_absolute_magnetisation_error = self.calculate_error(np.abs(np.array(magnetisations)))
    scaled_specific_heat_capacity = self.calculate_scaled_specific_heat_capacity(np.array(total_energies))
    susceptibility = self.calculate_susceptibility(np.abs(np.array(magnetisations)))
    
    # Divide the observables by the number of spins in the lattice
    # to give the values per spin.
    mean_total_energy_per_spin = mean_total_energy / (lattice_dimensions ** 2)
    mean_total_energy_error_per_spin = mean_total_energy_error /(lattice_dimensions ** 2)
    mean_absolute_magnetisation_per_spin = mean_absolute_magnetisation / (lattice_dimensions ** 2)
    mean_absolute_magnetisation_error_per_spin = mean_absolute_magnetisation_error / (lattice_dimensions ** 2)
    absolute_magnetisations_per_spin = np.abs(np.array(magnetisations)) / (lattice_dimensions ** 2)
    scaled_specific_heat_capacity_per_spin = scaled_specific_heat_capacity
    susceptibility_per_spin = susceptibility / (lattice_dimensions ** 2)
    
    absolute_magnetisations_per_spin_dict = {"Absolute Magnetisations Per Spin" : absolute_magnetisations_per_spin
    }
    absolute_magnetisations_per_spin_values = pd.DataFrame.from_dict(absolute_magnetisations_per_spin_dict)
    absolute_magnetisations_per_spin_values.to_csv("Absolute_Magnetisations_Per_Spin/absolute_magnetisations_per_spin_for_temperaterature_" + str(np.round(temperature, 2)) + ".txt")
    
    return mean_total_energy_per_spin, mean_total_energy_error_per_spin, mean_absolute_magnetisation_per_spin, mean_absolute_magnetisation_error_per_spin, scaled_specific_heat_capacity_per_spin, susceptibility_per_spin
    
  def calculate_observables_glauber_with_auto_correlation_times(self):
    """
    Calculate the observables of the Ising model using the
    auto-correlation times.
    """
    nsweeps = self.get_nsweeps()
    lattice_dimensions = self.get_lattice_dimensions()
    temperature = self.get_temperature()
    spin_lattice = self.get_spin_lattice()
    auto_correlation_time = self.get_auto_correlation_time()
    
    total_energies = []
    magnetisations = []

    for sweep in range(nsweeps+101):
      for row in range(lattice_dimensions):
        for column in range(lattice_dimensions):
          self.select_candidate_state_glauber()
          self.calculate_energy_difference_glauber()
          self.metropolis_algorithm_glauber()

      if sweep > 100 and not np.mod(sweep, auto_correlation_time):
        with open("Spins_Data/Spins_Data_" + str(np.round(temperature, 2)) + "/spins_glauber_" + str(np.round(temperature, 2)) + "_sweeps=" + str(sweep) + ".dat", "a") as file_object:
          for row in range(lattice_dimensions):
            for column in range(lattice_dimensions):
              file_object.write("%d %d %lf\n" % (row, column, spin_lattice[row, column]))
              
        # Calculate the mean total energy and magnetisation 
        # of the current state.
        current_energy = self.calculate_total_energy()
        current_magnetisation = self.calculate_magnetisation()
          
        # Append the current mean total energy and magnetisation
        # to the lists of mean total energies and magnetisations.
        total_energies.append(current_energy)
        magnetisations.append(current_magnetisation)
    
    mean_total_energy = np.mean(np.array(total_energies))
    mean_total_energy_error = self.calculate_error(np.array(total_energies))
    mean_absolute_magnetisation = np.mean(np.abs(np.array(magnetisations)))
    mean_absolute_magnetisation_error = self.calculate_error(np.abs(np.array(magnetisations)))
    scaled_specific_heat_capacity = self.calculate_scaled_specific_heat_capacity(np.array(total_energies))
    susceptibility = self.calculate_susceptibility(np.abs(np.array(magnetisations)))
    
    # Divide the observables by the number of spins in the lattice
    # to give the values per spin.
    mean_total_energy_per_spin = mean_total_energy / (lattice_dimensions ** 2)
    mean_total_energy_error_per_spin = mean_total_energy_error /(lattice_dimensions ** 2)
    mean_absolute_magnetisation_per_spin = mean_absolute_magnetisation / (lattice_dimensions ** 2)
    mean_absolute_magnetisation_error_per_spin = mean_absolute_magnetisation_error / (lattice_dimensions ** 2)
    absolute_magnetisations_per_spin = np.abs(np.array(magnetisations)) / (lattice_dimensions ** 2)
    scaled_specific_heat_capacity_per_spin = scaled_specific_heat_capacity / (lattice_dimensions ** 2)
    susceptibility_per_spin = susceptibility / (lattice_dimensions ** 2)
    
    absolute_magnetisations_per_spin_dict = {"Absolute Magnetisations Per Spin" : absolute_magnetisations_per_spin
    }
    absolute_magnetisations_per_spin_values = pd.DataFrame.from_dict(absolute_magnetisations_per_spin_dict)
    absolute_magnetisations_per_spin_values.to_csv("Absolute_Magnetisations_Per_Spin/absolute_magnetisations_per_spin_for_temperaterature_" + str(np.round(temperature, 2)) + ".txt")
    
    return mean_total_energy_per_spin, mean_total_energy_error_per_spin, mean_absolute_magnetisation_per_spin, mean_absolute_magnetisation_error_per_spin, scaled_specific_heat_capacity_per_spin, susceptibility_per_spin
  
  def calculate_observables_kawasaki(self):
    """
    Calculate the observables of the Ising model using
    Kawasaki dynamics.
    """
    nsweeps = self.get_nsweeps()
    lattice_dimensions = self.get_lattice_dimensions()
    temperature = self.get_temperature()
    spin_lattice = self.get_spin_lattice()
    total_energies = []
    
    for sweep in range(nsweeps + 101):
      for row in range(lattice_dimensions):
        for column in range(lattice_dimensions):
          self.select_candidate_state_kawasaki()
          self.calculate_energy_difference_kawasaki()
          self.metropolis_algorithm_kawasaki()
          
      if not np.mod(sweep, 10) and sweep > 100:
        # Calculate the mean total energy and magnetisation 
        # of the current state.
        current_energy = self.calculate_total_energy()
        
        # Append the current mean total energy and magnetisation
        # to the lists of mean total energies and magnetisations.
        total_energies.append(current_energy)
    
    mean_total_energy = np.mean(np.array(total_energies))
    mean_total_energy_error = self.calculate_error(np.array(total_energies))
    # write method to calculate error (jackknife or bootstrap)
    scaled_specific_heat_capacity = self.calculate_scaled_specific_heat_capacity(np.array(total_energies))
    
    # Divide observables by the number of spins in the
    # lattice to give the values per spin.
    mean_total_energy_per_spin = mean_total_energy / (lattice_dimensions ** 2)
    mean_total_energy_error_per_spin = mean_total_energy_error / (lattice_dimensions ** 2)
    scaled_specific_heat_capacity_per_spin = scaled_specific_heat_capacity / (lattice_dimensions ** 2)
      
    return mean_total_energy_per_spin, mean_total_energy_error_per_spin, scaled_specific_heat_capacity_per_spin
    
  def calculate_observables(self, dynamical_rule):
    """
    Calculate the observables of the Ising model
    for temperatures between 1.0 and 3.54 using
    either Glauber or Kawasaki dynamics.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    mean_total_energies_per_spin = []
    mean_total_energies_errors_per_spin = []
    mean_absolute_magnetisations_per_spin = []
    mean_absolute_magnetisations_errors_per_spin = []
    scaled_specific_heat_capacities_per_spin = []
    susceptibilities_per_spin = []
    with_auto_correlation_times_condition = self.get_with_auto_correlation_times_condition()
    
    if with_auto_correlation_times_condition:
      auto_correlation_times = pd.read_csv("Auto_Correlation_Times/auto_correlation_times.csv")["Auto Correlation Times"].to_numpy()
    for index, temperature in enumerate(np.round(np.linspace(1.0, 3.54, 21), 2)):
      self.set_temperature(temperature)
      if with_auto_correlation_times_condition:
        self.set_auto_correlation_time(auto_correlation_times[index])
        print(auto_correlation_times[index])
      if dynamical_rule == "Glauber":
        if with_auto_correlation_times_condition:
          mean_total_energy_per_spin, mean_total_energy_error_per_spin, mean_absolute_magnetisation_per_spin, mean_absolute_magnetisation_error_per_spin, scaled_specific_heat_capacity_per_spin, susceptibility_per_spin = self.calculate_observables_glauber_with_auto_correlation_times()
        else:
          mean_total_energy_per_spin, mean_total_energy_error_per_spin, mean_absolute_magnetisation_per_spin, mean_absolute_magnetisation_error_per_spin, scaled_specific_heat_capacity_per_spin, susceptibility_per_spin = self.calculate_observables_glauber_without_auto_correlation_times()
        
        mean_total_energies_per_spin.append(mean_total_energy_per_spin)
        mean_total_energies_errors_per_spin.append(mean_total_energy_error_per_spin)
        mean_absolute_magnetisations_per_spin.append(mean_absolute_magnetisation_per_spin)
        mean_absolute_magnetisations_errors_per_spin.append(mean_absolute_magnetisation_error_per_spin)
        scaled_specific_heat_capacities_per_spin.append(scaled_specific_heat_capacity_per_spin)
        susceptibilities_per_spin.append(susceptibility_per_spin)

      elif dynamical_rule == "Kawasaki":
          mean_total_energy_per_spin, mean_total_energy_error_per_spin, scaled_specific_heat_capacity_per_spin = self.calculate_observables_kawasaki()
          
          mean_total_energies_per_spin.append(mean_total_energy_per_spin)
          mean_total_energies_errors_per_spin.append(mean_total_energy_error_per_spin)
          scaled_specific_heat_capacities_per_spin.append(scaled_specific_heat_capacity_per_spin)
          
    if dynamical_rule == "Glauber":
      observables_dictionary = {"Temperatures": np.round(np.linspace(1.0, 3.54, 21), 2),
                              "Mean Total Energies Per Spin" : mean_total_energies_per_spin,
                              "Mean Total Energies Errors Per Spin" : mean_total_energies_errors_per_spin,
                              "Mean Absolute Magnetisations Per Spin" : mean_absolute_magnetisations_per_spin,
                              "Mean Absolute Magnetisations Errors Per Spin" : mean_absolute_magnetisations_errors_per_spin,
                              "Scaled Specific Heat Capacities Per Spin" : scaled_specific_heat_capacities_per_spin,
                              "Susceptibilities Per Spin" : susceptibilities_per_spin,
                              }

      observables_dataframe = pd.DataFrame.from_dict(observables_dictionary)
      observables_dataframe.to_csv("Glauber_Observables/glauber_observables_data.csv")
      
    elif dynamical_rule == "Kawasaki":
      observables_dictionary = {"Temperatures": np.linspace(1.0, 3.54, 21),
                              "Mean Total Energies Per Spin": mean_total_energies_per_spin,
                              "Mean Total Energies Errors Per Spin": mean_total_energies_errors_per_spin,
                              "Scaled Specific Heat Capacities Per Spin": scaled_specific_heat_capacities_per_spin,
                              }
      observables_dataframe = pd.DataFrame.from_dict(observables_dictionary)
      observables_dataframe.to_csv("Kawasaki_Observables/kawasaki_observables_data.csv")
      
  def plot_mean_energy_against_temperature(self, dynamical_rule):
    """Plot the mean energy against temperature."""
    lattice_dimensions = self.get_lattice_dimensions()
    temperatures = []
    mean_energies_per_spin = []
    mean_energies_errors_per_spin = []
    if dynamical_rule == "Glauber":
      observables_data = pd.read_csv("Glauber_Observables/glauber_observables_data.csv")
      temperatures = observables_data["Temperatures"]
      mean_energies_per_spin = observables_data["Mean Total Energies Per Spin"]
      mean_energies_errors_per_spin = observables_data["Mean Total Energies Errors Per Spin"]
    elif dynamical_rule == "Kawasaki":
      observables_data = pd.read_csv("Kawasaki_Obervables/kawasaki_observables_data.csv")
      temperatures = observables_data["Temperatures"]
      mean_energies_per_spin = observables_data["Mean Total Energies Per Spin"]
      mean_energies_errors_per_spin = observables_data["Mean Total Energies Errors Per Spin"]
    
    plt.cla()
    plt.title("Mean Energy Per Spin Against Temperature For " + dynamical_rule + " Dynamics")
    plt.xlabel('Temperature, T')
    plt.ylabel('Mean Energy Per Spin, $\displaystyle\langle E \\rangle$')
    plt.errorbar(temperatures, mean_energies_per_spin, yerr=mean_energies_errors_per_spin)
    plt.savefig("Graphs/mean_energy_per_spin_against_temperature.png")
    plt.show()
    
  def plot_absolute_magnetisation_against_temperature(self):
    """Plot the mean energy against temperature."""
    lattice_dimensions = self.get_lattice_dimensions()
    observables_data = pd.read_csv("Glauber_Observables/glauber_observables_data.csv")
    temperatures = observables_data["Temperatures"]
    absolute_mean_magnetisations_per_spin = observables_data["Mean Absolute Magnetisations Per Spin"]
    absolute_mean_magnetisations_errors_per_spin = np.abs(observables_data["Mean Absolute Magnetisations Errors Per Spin"])
    
    plt.cla()
    plt.title('Absolute Mean Magnetisation Per Spin Against Temperature For Glauber Dynamics')
    plt.xlabel('Temperature, T')
    plt.ylabel('Absolute Mean Magnetisation Per Spin, $\displaystyle\langle |M| \\rangle$')
    plt.errorbar(temperatures, absolute_mean_magnetisations_per_spin, yerr=absolute_mean_magnetisations_errors_per_spin)
    plt.savefig("Graphs/absolute_mean_magnetisation_against_temperature.png")
    plt.show()
    
  def plot_scaled_specific_heat_against_temperature(self, dynamical_rule):
    """Plot the mean energy against temperature."""
    lattice_dimensions = self.get_lattice_dimensions()
    temperatures = []
    scaled_specific_heat_capacities = []
    mean_energies_errors = []
    if dynamical_rule == "Glauber":
      observables_data = pd.read_csv("Glauber_Observables/glauber_observables_data.csv")
      temperatures = observables_data["Temperatures"]
      scaled_specific_heat_capacities_per_spin = observables_data["Scaled Specific Heat Capacities Per Spin"]
      mean_energies_errors_per_spin = observables_data["Mean Total Energies Errors Per Spin"]
    elif dynamical_rule == "Kawasaki":
      observables_data = pd.read_csv("Kawasaki_Obervables/kawasaki_observables_data.csv")
      temperatures = observables_data["Temperatures"]
      scaled_specific_heat_capacities_per_spin = observables_data["Scaled Specific Heat Capacities Per Spin"]
      mean_energies_errors_per_spin = observables_data["Mean Total Energies Errors Per Spin"]
    
    plt.cla()
    plt.title("Scaled Specific Heat Capacity against Temperature for " + dynamical_rule + " Dynamics")
    plt.xlabel('Temperature, $T$')
    plt.ylabel('Scaled Specific Heat Capacity, $c$')
    plt.errorbar(temperatures, scaled_specific_heat_capacities_per_spin, yerr=mean_energies_errors_per_spin)
    plt.savefig("Graphs/scaled_specific_heat_against_temperature.png")
    plt.show()
    
  def plot_susceptibility_against_temperature(self):
    """Plot the mean energy against temperature."""
    lattice_dimensions = self.get_lattice_dimensions()
    observables_data = pd.read_csv("Glauber_Observables/glauber_observables_data.csv")
    temperatures = observables_data["Temperatures"]
    susceptibilities_per_spin = observables_data["Susceptibilities Per Spin"]
    absolute_mean_magnetisations_errors_per_spin = np.abs(observables_data["Mean Absolute Magnetisations Errors Per Spin"])
    
    plt.cla()
    plt.title("Susceptibility Per Spin Against Temperature For Glauber Dynamics")
    plt.xlabel('Temperature, $T$')
    plt.ylabel('Susceptibilities Per Spin, $\\frac{\chi}{N^2}$')
    plt.errorbar(temperatures, susceptibilities_per_spin, yerr=absolute_mean_magnetisations_errors_per_spin)
    plt.savefig("Graphs/susceptibilities_per_spin_against_temperature.png")
    plt.show()
    
  def plot_auto_correlation_function_of_magnetisation_per_spin_against_sweeps(self, tau):
    """
    Plot the absolute magnetisation against the 
    sweeps.
    """
    lattice_dimensions = self.get_lattice_dimensions()
    nsweeps = self.get_nsweeps()
    temperature = self.get_temperature()
    sweeps = np.arange(0, nsweeps)
    plt.cla()
    plt.title('Auto-Correlation Function of Magnetisation Per Spin Against the Number of Sweeps For Glauber Dynamics')
    plt.xlabel('Sweeps')
    plt.ylabel('Auto-Correlation Function of Magnetisation Per Spin, $\displaystyle\langle |M| \\rangle$')
    plt.scatter(sweeps, self.auto_correlation_time_exponential(sweeps, tau))
    plt.savefig("Graphs/Auto_Correlation_Function_of_Magnetisations_Per_Spin_Against_Sweeps/magnetisation_against_sweeps_at_temperature_" + str(np.round(temperature, 2)) + ".png")
    
  def calculate_correlation_function(self, temperature):
    """Calculate the correlation function values."""
    nsweeps = int(self.get_nsweeps())
    
    absolute_magnetisations_per_spin_data = pd.read_csv("Absolute_Magnetisations_Per_Spin/absolute_magnetisations_per_spin_for_temperaterature_" + str(round(temperature, 2)) + ".txt")
    absolute_magnetisations_per_spin = absolute_magnetisations_per_spin_data["Absolute Magnetisations Per Spin"]
    mean_absolute_magnetisation = np.mean(np.array(absolute_magnetisations_per_spin))
    correlation_values = []
    for sweep in range(1, nsweeps):
      correlation_values.append((absolute_magnetisations_per_spin[sweep - 1] * absolute_magnetisations_per_spin[sweep]) - mean_absolute_magnetisation ** 2)
    
    return np.abs(np.array(correlation_values))
  
  def auto_correlation_time_exponential(self, sweeps, tau):
    """
    The exponential distribution representing the
    auto-correlation function.
    """
    return np.exp(-sweeps/tau)
    
  def calculate_auto_correlation_time(self):
    """
    Calculate the auto-correlation time by fitting the auto-correlation
    function.
    """
    nsweeps = int(self.get_nsweeps())
    sweeps = np.arange(0, nsweeps-1)
    auto_correlation_times = []
    for temperature in np.linspace(1.00, 3.54, 21):
      correlation_values = self.calculate_correlation_function(temperature)
      popt = curve_fit(self.auto_correlation_time_exponential, sweeps, correlation_values)[0]
      tau = int(abs(1 / popt))
      self.plot_auto_correlation_function_of_magnetisation_per_spin_against_sweeps(tau)
      auto_correlation_times.append(tau)
      
    auto_correlation_times_dict = {"Auto Correlation Times": auto_correlation_times}
    auto_correlation_times_data = pd.DataFrame.from_dict(auto_correlation_times_dict)
    auto_correlation_times_data.to_csv("Auto_Correlation_Times/auto_correlation_times.csv")
