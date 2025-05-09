# Aircraft Engine Fault Detection (PGM)

This project implements a Dynamic Bayesian Network (DBN) and Markov Random Field (MRF) framework for detecting early-stage faults in aircraft engines using simulated sensor data.

## Project Structure

- `Data_Gen/`: Simulation and discretization code
- `DBN/`: DBN structure, inference models, and visualizations
- `PreProcessing/`: MRF smoothing logic
- `Utils/`: Baselines, evaluation, and utility functions
- `Data/`: Generated datasets, experiment results, plots
- `run_experiment.py`: Main pipeline runner
- `evaluation.py`: Outputs model metrics and confusion matrices
- `environment.yaml`: Conda environment dependencies
- `PGM_GUI_Viewer/`: Lightweight Flask GUI for report visualization

## How to Run

1. Create the environment:
   ```
   conda env create -f environment.yaml
   conda activate aircraft_env
   ```

2. Run experiments:
   ```
   python run_experiment.py
   ```

3. Evaluate:
   ```
   cd Utils/
   python evaluation.py
   ```

4. Launch GUI (optional):
   ```
   cd PGM_GUI_Viewer/
   python viewer_app.py
   ```

## Report

See `2561034_report.pdf` for detailed results and discussion.
