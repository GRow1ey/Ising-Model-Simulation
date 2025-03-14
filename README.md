# Advanced Simulation of the Ising Model with Neural Network Integration

**Description:** This project involves the simulation of the Ising model, a fundamental mathematical model of ferromagnetism, utilizing both Glauber and Kawasaki dynamics. The simulation is designed to calculate various physical observables and plot them against temperature, providing a comprehensive analysis of the system's behavior.

The codebase is meticulously organized into several Python scripts, with a central class, `SpinLattice`, that manages the core simulation logic. This modular approach ensures clarity and ease of maintenance.

## Key Features:

* Dynamic Simulation: Implementation of both Glauber and Kawasaki dynamics to simulate the Ising model.
* Comprehensive Analysis: Calculation and plotting of physical observables against temperature to study the system's properties.
* Modular Codebase: Organized Python scripts with a main class, `SpinLattice`, for streamlined simulation management.

**Future Enhancements:** I plan to enhance this project by integrating machine learning techniques. Specifically, I aim to train a neural network on the simulation data to perform image classification. This neural network will be capable of identifying the temperature of spin particles within the simulation based on grids of black and white squares representing spin particles.

## Technologies Used:

* Python
* Matplotlib (for plotting)
* NumPy (for numerical calculations)
* TensorFlow/PyTorch (for future neural network integration)
  
This project not only demonstrates my proficiency in computational physics and Python programming but also showcases my ability to integrate advanced machine learning techniques to enhance simulation analysis.

## Folder Structure

```
.gitignore
.sfdx/
    tools/
Absolute_Magnetisations_Per_Spin/
    absolute_magnetisations_per_spin_for_temperaterature_1.0.txt
    absolute_magnetisations_per_spin_for_temperaterature_1.13.txt
    absolute_magnetisations_per_spin_for_temperaterature_1.25.txt
    ...
Auto_Correlation_Times/
    auto_correlation_times.csv
Glauber_Observables/
    ...
Graphs/
ising_model_animation.py
ising_model_calculate_autocorrelation_times.py
ising_model_calculate_observables.py
ising_model_interface.py
README.md
spin_lattice.py
```

## Files

- `ising_model_animation.py`: Script to animate the Ising model using Glauber or Kawasaki dynamics.
- `ising_model_calculate_autocorrelation_times.py`: Script to calculate the auto-correlation times for the Ising model.
- `ising_model_calculate_observables.py`: Script to calculate various observables of the Ising model.
- `ising_model_interface.py`: Interface script to interact with the Ising model simulation.
- `spin_lattice.py`: Contains the `SpinLattice` class which implements the core logic for the Ising model simulation.

## Usage

### Animation

To animate the Ising model, run:

```sh
python3 ising_model_animation.py N T Glauber/Kawasaki
```

- `N`: Lattice dimensions (e.g., 10 for a 10x10 lattice).
- `T`: Temperature.
- `Glauber/Kawasaki`: Dynamics type.

### Calculate Auto-Correlation Times

To calculate the auto-correlation times, run:

```sh
python3 ising_model_calculate_autocorrelation_times.py N Glauber/Kawasaki nsweeps
```

- `N`: Lattice dimensions.
- `Glauber/Kawasaki`: Dynamics type.
- `nsweeps`: Number of sweeps.

### Calculate Observables

To calculate the observables, run:

```sh
python3 ising_model_calculate_observables.py N Glauber/Kawasaki nsweeps with/without_auto-correlation_times
```

- `N`: Lattice dimensions.
- `Glauber/Kawasaki`: Dynamics type.
- `nsweeps`: Number of sweeps.
- `with/without_auto-correlation_times`: Whether to use auto-correlation times.

## Class `SpinLattice`

The `SpinLattice` class in [`spin_lattice.py`](spin_lattice.py) handles the core simulation logic. Key methods include:

- `select_candidate_state_glauber()`: Selects a candidate state for Glauber dynamics.
- `calculate_auto_correlation_time()`: Calculates the auto-correlation time.
- `calculate_observables_glauber_with_auto_correlation_times()`: Calculates observables using Glauber dynamics with auto-correlation times.
- `calculate_observables_glauber_without_auto_correlation_times()`: Calculates observables using Glauber dynamics without auto-correlation times.
- `calculate_observables_kawasaki()`: Calculates observables using Kawasaki dynamics.
- `plot_mean_energy_against_temperature()`: Plots mean energy against temperature.
- `plot_absolute_magnetisation_against_temperature()`: Plots absolute magnetisation against temperature.
- `plot_scaled_specific_heat_against_temperature()`: Plots scaled specific heat against temperature.
- `plot_susceptibility_against_temperature()`: Plots susceptibility against temperature.

## Data Files

- `Absolute_Magnetisations_Per_Spin/`: Contains data files for absolute magnetisations per spin at various temperatures.
- `Auto_Correlation_Times/auto_correlation_times.csv`: Contains auto-correlation times data.
- `Glauber_Observables/`: Contains data files for observables calculated using Glauber dynamics.
- `Graphs/`: Contains generated plots.

## Dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`

Install the dependencies using:

```sh
pip install numpy matplotlib pandas scipy
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
