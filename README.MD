# Genetic Algorithm for Text String Evolution

**Authors:** Norbert Klockiewicz, Kacper Majchrzak

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
