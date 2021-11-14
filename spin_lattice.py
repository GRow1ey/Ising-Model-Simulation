import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True

class SpinLattice():
  """A class to represent the lattice of spins in the Ising Model."""
  
  def __init__(self, lattice_dimensions, temperature=1.0, J=1.0, nsweeps=10000):
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
    self.initial_state = True
    
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
    """Method to return J."""
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
    """Method to update thhe jtrial_1 attribute."""
    self.jtrial_1 = jtrial_1
    
  def get_initial_state(self):
    """Method to return the initial state attribute."""
    return self.initial_state
  
  def update_initial_state(self):
    """
    Method to change the initial state to false after the 
    first iteration of calculating observables.
    """
    self.initial_state = False
      
  def initialise_spin_lattice_kawasaki(self):
        """
        Initialise the spin lattice for Kawasaki dynamics
        to allow accurate calculation of observables.
        Half of the spins are initialised as up and half as down.
        This is not used for animating the Ising model.
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
  
  def calculate_mean_total_energy(self):
    """Calculate the mean total energy of the lattice of spins."""
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
    """Select a candidate state."""
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
    """"""
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
    """"""
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    
    # Calculate the difference between the
    # candidate state and the current state.
    energy_current_state = self.calculate_spin_configuration_energy(itrial, jtrial)
    
    energy_candidate_state = -energy_current_state
    
    return energy_candidate_state - energy_current_state
  
  def nearest_neighbours_check_kawasaki(self):
    """"""
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
    """"""
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
    """"""
    spin_lattice = self.get_spin_lattice()
    energy_difference = self.calculate_energy_difference_glauber()
    boltzmann_weight = self.calculate_boltzmann_weight(energy_difference)
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    
    random_number = np.random.random()
    if energy_difference <= 0:
      spin_lattice[itrial, jtrial] *= -1
    elif random_number <= boltzmann_weight:
      spin_lattice[itrial, jtrial] *= -1
      
  def metropolis_algorithm_kawasaki(self):
    """"""
    spin_lattice = self.get_spin_lattice()
    energy_difference = self.calculate_energy_difference_kawasaki()
    boltzmann_weight = self.calculate_boltzmann_weight(energy_difference)
    itrial = self.get_itrial()
    jtrial = self.get_jtrial()
    itrial_1 = self.get_itrial_1()
    jtrial_1 = self.get_jtrial_1()
    
    random_number = np.random.random()
    if energy_difference <= 0:
      spin_lattice[itrial, jtrial] *= -1
      spin_lattice[itrial_1, jtrial_1] *= -1
    elif random_number <= boltzmann_weight:
      spin_lattice[itrial, jtrial] *= -1
      spin_lattice[itrial_1, jtrial_1] *= -1
      
  def animate_ising_model_glauber(self):
    """"""
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
        
  def animate_ising_model_kawasaki(self):
    """"""
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
    
  def calculate_observables_glauber(self):
    """"""
    nsweeps = self.get_nsweeps()
    lattice_dimensions = self.get_lattice_dimensions()
    temperature = self.get_temperature()
    spin_lattice = self.get_spin_lattice()
    initial_state = self.get_initial_state()
    
    mean_total_energies = []
    magnetisations = []
    
    if initial_state:
      # Calculate the mean total energy and magnetisation of
      # the initial lattice.
      current_energy = self.calculate_mean_total_energy()
      current_magnetisation = self.calculate_magnetisation()
      
      # Append the initial mean total energy and magnetisation,
      # to the list of mean total energies and magnetisations.
      mean_total_energies.append(current_energy)
      magnetisations.append(current_magnetisation)
      
      self.update_initial_state()
      
    else:
      # Set the previous spin lattice equal to the spin
      # lattice used in the previous calculation of
      # observables if this is not the initial state.
      self.set_previous_spin_lattice(spin_lattice)
    
    for sweep in range(nsweeps):
      for row in range(lattice_dimensions):
        for column in range(lattice_dimensions):
          self.select_candidate_state_kawasaki()
          self.calculate_energy_difference_kawasaki()
          self.metropolis_algorithm_kawasaki()
          
      if not np.mod(sweep, 10):
        # Calculate the mean total energy and magnetisation 
        # of the current state.
        current_energy = self.calculate_mean_total_energy()
        current_magnetisation = self.calculate_magnetisation()
        
        # Append the current mean total energy and magnetisation
        # to the lists of mean total energies and magnetisations.
        mean_total_energies.append(current_energy)
        magnetisations.append(current_magnetisation)
        
    # Store the mean total energies and magnetisations in a data
    # file.
    with open("glauber_observables_data" + str(temperature) + ".dat") as file_object:
      file_object.write(mean_total_energies)
      file_object.write(magnetisations)
      
      
        
    
    
    
    
  def calculate_observables_kawasaki(self):
    """"""
    
  def calculate_observables(self, dynamical_rule):
    """"""
    lattice_dimensions = self.get_lattice_dimensions()
    susceptibilities = []
    energies = []
    scaled_heat_capacities = []
    absolute_magnetisation = []
    
    for temperature in np.arange(1, 3, 0.1):
      self.set_temperature(temperature)
      if dynamical_rule == "Glauber":
          self.calculate_observables_glauber()
      elif dynamical_rule == "Kawasaki":
          self.calculate_observables_kawasaki()
      
      