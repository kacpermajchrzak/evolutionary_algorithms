# *Using Genetic Algorithms to Evolve Text Strings Toward a Target Phrase*

**Authors:** Norbert Klockiewicz, Kacper Majchrzak

The goal of this project is to demonstrate how a **genetic algorithm (GA)** can be used to gradually transform a random string of characters into a chosen target phrase (for example, *“HELLO WORLD”*). The idea is based on the principles of biological evolution: selection, crossover, and mutation.

## **How the Algorithm Works**

1. Start with a population of random character strings.
2. Measure how good each string is by comparing it to the target phrase.
3. Select the better strings to act as parents.
4. Mix parts of parent strings together (**crossover**) to create new offspring.
5. Randomly change some characters (**mutation**) to keep diversity in the population.
6. Repeat this process for multiple generations until the exact target phrase appears.

## **What We Will Compare**

To understand how different GA settings affect performance, we will test:

* Different **mutation methods** (simple random replacement, adaptive mutation, Gaussian mutation, and swap mutation).
* Various **crossover strategies** (single-point, two-point, uniform, and blended crossover).
* Different **selection approaches** (tournament, roulette, and best-individual selection).

We will observe:

* How many generations it takes to reach the target.
* How the average and best fitness change over time.
* Whether the population gets stuck (premature convergence) or continues improving.

## **Expected Results**

* Mutation that adapts over time usually performs better because the algorithm explores more early on and fine-tunes later.
* Crossover strategies that mix characters more evenly (like uniform crossover) tend to help maintain diversity.
* Tournament selection is expected to provide stable progress without reducing variety too quickly.

## **Outcome**

By the end of the project, we should have:

* A clear comparison of different GA settings.
* Visual charts showing evolution progress.
* A better understanding of how to control exploration vs. refinement in evolutionary algorithms.

The project demonstrates how **simple evolutionary rules can lead to complex, meaningful results**.
