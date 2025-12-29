# Genetic Algorithm for Text String Evolution

**Authors:** Norbert Klockiewicz, Kacper Majchrzak

A comprehensive implementation of genetic algorithms to evolve random character strings toward a target phrase. This project demonstrates the principles of biological evolution (selection, crossover, mutation) applied to computational problem-solving.

## Overview

This project uses DEAP (Distributed Evolutionary Algorithms in Python) to implement and compare different genetic algorithm configurations. The goal is to transform a random string of characters into a chosen target phrase (e.g., "HELLO WORLD").

## Division of Labor

The project will be developed collaboratively by Kacper Majchrzak and Norbert Klockiewicz.
Both contributors will work on most parts of the implementation, with responsibilities overlapping depending on progress and individual strengths.
The division below serves primarily as an indicative outline rather than a strict separation of tasks.

**Norbert Klockiewicz** (main focus):

- Implementation of key genetic algorithm components (mutation, crossover, and selection operators)
- Development of fitness evaluation and population management functions
- Experiment management and configuration system
- Statistical analysis and result interpretation
- Testing, validation, and documentation

**Kacper Majchrzak** (main focus):

- Implementation of key genetic algorithm components (mutation, crossover, and selection operators)
- Visualization and plotting functionality using matplotlib
- Data export and file handling (CSV, JSON output)
- Command-line interface and user interaction design
- Testing, validation, and documentation

Note:
The division of work is flexible and may change as the project progresses. Both team members will actively contribute to all major parts of the system, ensuring mutual code review, idea exchange, and shared responsibility for the final result.

## Input Data Description

The program will operate with minimal input requirements and will generate most data internally:

- **Target String:** A text phrase that will be provided as a command-line argument (default: "HELLO WORLD")  
  - It will be able to contain any printable ASCII characters  
  - Its length will determine the chromosome size for the genetic algorithm  
  - Examples: "HELLO WORLD", "TO BE OR NOT TO BE", "GENETIC ALGORITHMS"

- **Algorithm Selection Parameters:**  
  - Mutation method: `simple`, `gaussian`, `swap`, `adaptive`  
  - Crossover method: `one_point`, `two_point`, `uniform`, `blend`  
  - Selection method: `tournament`, `roulette`, `best`

**No external data files will be required** – the program will generate random initial populations and evolve them toward the target string using only the provided parameters.

## Features

- **Multiple Mutation Methods:**
  - Simple random replacement
  - Gaussian mutation (position-based)
  - Swap mutation (character swapping)
  - Adaptive mutation (decreases over generations)

- **Various Crossover Strategies:**
  - Single-point crossover
  - Two-point crossover
  - Uniform crossover
  - Blend crossover

- **Different Selection Approaches:**
  - Tournament selection
  - Roulette wheel selection
  - Best individual selection

- **Comprehensive Statistics Tracking:**
  - Best fitness over time
  - Average fitness progression
  - Population diversity metrics
  - Standard deviation of fitness

- **Visualization and Analysis:**
  - Evolution progress charts
  - Configuration comparison graphs
  - Data export to CSV and JSON

## Installation

1. Clone or download this repository

2. Install dependencies:

With uv:

```bash
uv sync
```

Or install manually:

```bash
pip install deap matplotlib
```

## Usage

### Default Mode (Comprehensive Experiment)

Simply run the program without arguments to execute a comprehensive comparison:

```bash
python main.py
```

This will:

- Test all combinations of mutation, crossover, and selection methods (48 total configurations)
- Run each configuration 2 times by default
- Show the top 2 best performing configurations
- Generate detailed plots and data for the top configurations
- Save all results to organized directories

**Customize the experiment:**

```bash
python main.py experiment "HELLO WORLD" 3 5
```

Parameters:

- Target string (default: "HELLO WORLD")
- Number of runs per configuration (default: 2)
- Number of top configurations to analyze (default: 2)

### Demo Mode

Run a quick demonstration with 3 pre-selected configurations:

```bash
python main.py demo
```

This will:

- Run 3 different GA configurations
- Generate visualization plots (PNG files)
- Export results to CSV and JSON files
- Complete quickly for testing purposes

### Single Custom Run

Run a single configuration with custom parameters:

```bash
python main.py single "HELLO WORLD" adaptive uniform tournament
```

Parameters:

- Target string (default: "HELLO WORLD")
- Mutation method: `simple`, `gaussian`, `swap`, `adaptive`
- Crossover method: `one_point`, `two_point`, `uniform`, `blend`
- Selection method: `tournament`, `roulette`, `best`

## Output Files

The program organizes output files into three directories:

- **`plots/`** - Visualization files (PNG images)
- **`data/`** - Detailed CSV data files
- **`results/`** - Summary JSON files

### Generated by Default Mode (Comprehensive Experiment)

**Plots (`plots/` directory):**

- `top_N_configurations.png` - Comparison bar chart of top N configurations
- `rank_1_config.png` - Detailed evolution plot for best configuration
- `rank_2_config.png` - Detailed evolution plot for 2nd best configuration
- (etc. for each top configuration)

**Data (`data/` directory):**

- `rank_1_config.csv` - Generation-by-generation stats for best configuration
- `rank_2_config.csv` - Generation-by-generation stats for 2nd best configuration
- (etc. for each top configuration)

**Results (`results/` directory):**

- `comprehensive_experiment.json` - Complete results for all 48 configurations

### Generated by Demo Mode

**Plots (`plots/` directory):**

- `evolution_config1.png` - Evolution progress for configuration 1
- `evolution_config2.png` - Evolution progress for configuration 2
- `evolution_config3.png` - Evolution progress for configuration 3
- `comparison.png` - Side-by-side comparison of configurations

**Data (`data/` directory):**

- `results_config1.csv` - Detailed statistics for configuration 1
- `results_config2.csv` - Detailed statistics for configuration 2
- `results_config3.csv` - Detailed statistics for configuration 3

**Results (`results/` directory):**

- `demo_summary.json` - Summary of all demo runs

### Generated by Single Mode

**Plots (`plots/` directory):**

- `custom_run.png` - Evolution progress visualization

**Data (`data/` directory):**

- `custom_run.csv` - Detailed generation-by-generation statistics

**Results (`results/` directory):**

- `custom_run.json` - Summary of the run

## Key Metrics

The program tracks and reports:

- **Generations to Target:** How many generations needed to reach the target
- **Best Fitness:** Highest fitness score in the population
- **Average Fitness:** Mean fitness across all individuals
- **Diversity:** Percentage of unique individuals in the population
- **Standard Deviation:** Fitness variance in the population
- **Success Rate:** Percentage of runs that reached the target (experiment mode)

## Dependencies

- Python 3.13+
- DEAP (Distributed Evolutionary Algorithms in Python)
- matplotlib (for visualization)
