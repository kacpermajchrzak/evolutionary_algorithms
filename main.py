import random
import string
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from deap import base, creator, tools
import matplotlib.pyplot as plt
import sys

# ---------- Directory Setup ----------
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ---------- Configuration ----------
TARGET = "HELLO WORLD"
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
CHARS = string.ascii_letters + " "  # character pool

# ---------- DEAP setup ----------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    # Individual must be a mutable container (list) so operators can modify in-place
    creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore


def create_individual(target_len):
    """Return a list of random chars of length target_len."""
    return creator.Individual(random.choice(CHARS) for _ in range(target_len))  # type: ignore


def evaluate_fitness(individual, target):
    """Fitness = number of matching characters to target."""
    # individual is a list of chars
    fitness = sum(a == b for a, b in zip(individual, target))
    return (fitness,)


# ---------- Mutation operators (in-place) ----------
def mutate_simple(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(CHARS)
    return (individual,)


def mutate_gaussian(individual, mutation_rate, sigma=0.2):
    # choose a number of positions relative to sigma and mutate them
    n = max(1, int(len(individual) * sigma))
    positions = random.sample(range(len(individual)), n)
    for pos in positions:
        if random.random() < mutation_rate:
            individual[pos] = random.choice(CHARS)
    return (individual,)


def mutate_swap(individual, mutation_rate):
    # perform several random swaps based on mutation_rate
    n_swaps = sum(1 for _ in range(len(individual)) if random.random() < mutation_rate)
    for _ in range(n_swaps):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return (individual,)


def mutate_adaptive(individual, base_rate, generation, max_gen=1000):
    adaptive_rate = base_rate * (1.0 - (generation / float(max_gen)))
    for i in range(len(individual)):
        if random.random() < adaptive_rate:
            individual[i] = random.choice(CHARS)
    return (individual,)


# ---------- Crossover operators (in-place) ----------
def cx_one_point(ind1, ind2):
    if len(ind1) <= 1:
        return ind1, ind2
    pt = random.randint(1, len(ind1) - 1)
    # swap tails
    tail1 = ind1[pt:]
    tail2 = ind2[pt:]
    ind1[pt:], ind2[pt:] = tail2, tail1
    return ind1, ind2


def cx_two_point(ind1, ind2):
    if len(ind1) < 2:
        return ind1, ind2
    p1 = random.randint(1, len(ind1) - 2)
    p2 = random.randint(p1 + 1, len(ind1) - 1)
    seg1 = ind1[p1:p2]
    seg2 = ind2[p1:p2]
    ind1[p1:p2], ind2[p1:p2] = seg2, seg1
    return ind1, ind2


def cx_uniform(ind1, ind2, prob=0.5):
    for i in range(len(ind1)):
        if random.random() < prob:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


# simple blend-like (same as uniform but with 50/50)
def cx_blend(ind1, ind2):
    return cx_uniform(ind1, ind2, prob=0.5)


# ---------- Main evolution function ----------
def run_evolution(
    target=TARGET,
    population_size=POPULATION_SIZE,
    generations=GENERATIONS,
    mutation_rate=MUTATION_RATE,
    crossover_rate=CROSSOVER_RATE,
    mutation_method="simple",
    crossover_method="one_point",
    selection_method="tournament",
    track_stats=True,
    verbose=True,
):
    """
    Run genetic algorithm evolution to match a target string.

    Returns:
        dict: Contains 'generations_to_target', 'stats_history', and other metrics
    """
    target = str(target)
    target_len = len(target)
    max_fitness = target_len

    # Statistics tracking
    stats_history = {
        "generation": [],
        "best_fitness": [],
        "avg_fitness": [],
        "std_fitness": [],
        "diversity": [],
        "best_string": [],
    }

    toolbox = base.Toolbox()

    # attributes / individual / population
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,  # type: ignore
        lambda: create_individual(target_len),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore
    toolbox.register("evaluate", evaluate_fitness, target=target)

    # selection
    if selection_method == "tournament":
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif selection_method == "roulette":
        toolbox.register("select", tools.selRoulette)
    elif selection_method == "best":
        toolbox.register("select", tools.selBest)
    else:
        toolbox.register("select", tools.selTournament, tournsize=3)

    # mating registration
    if crossover_method == "one_point":
        toolbox.register("mate", cx_one_point)
    elif crossover_method == "two_point":
        toolbox.register("mate", cx_two_point)
    elif crossover_method == "uniform":
        toolbox.register("mate", cx_uniform)
    elif crossover_method == "blend":
        toolbox.register("mate", cx_blend)
    else:
        toolbox.register("mate", cx_one_point)

    # mutation registration (for adaptive we will re-register inside loop)
    if mutation_method == "simple":
        toolbox.register("mutate", mutate_simple, mutation_rate=mutation_rate)
    elif mutation_method == "gaussian":
        toolbox.register(
            "mutate", mutate_gaussian, mutation_rate=mutation_rate, sigma=0.2
        )
    elif mutation_method == "swap":
        toolbox.register("mutate", mutate_swap, mutation_rate=mutation_rate)
    elif mutation_method == "adaptive":
        # placeholder; will be re-registered each generation with current generation
        toolbox.register(
            "mutate",
            mutate_adaptive,
            base_rate=mutation_rate,
            generation=0,
            max_gen=generations,
        )
    else:
        toolbox.register("mutate", mutate_simple, mutation_rate=mutation_rate)

    # initialize population and compute fitnesses
    pop = toolbox.population(n=population_size)  # type: ignore
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)  # type: ignore

    if verbose:
        print(f"\nTarget: '{target}'")
        print(f"Population Size: {population_size} | Generations: {generations}")
        print(
            f"Mutation: {mutation_method} ({mutation_rate}) | Crossover: {crossover_method} ({crossover_rate}) | Selection: {selection_method}"
        )
        print("-" * 80)

    generations_to_target = None

    for gen in range(1, generations + 1):
        # If adaptive mutation, update registration with current generation
        if mutation_method == "adaptive":
            # re-register mutate wrapping current generation
            toolbox.register(
                "mutate",
                mutate_adaptive,
                base_rate=mutation_rate,
                generation=gen,
                max_gen=generations,
            )

        # selection
        offspring = toolbox.select(pop, len(pop))  # type: ignore
        offspring = list(map(toolbox.clone, offspring))  # type: ignore

        # apply crossover
        for i in range(1, len(offspring), 2):
            if random.random() < crossover_rate:
                toolbox.mate(offspring[i - 1], offspring[i])  # type: ignore
                # invalidate fitness
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # apply mutation
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                toolbox.mutate(offspring[i])  # type: ignore
                try:
                    del offspring[i].fitness.values
                except AttributeError:
                    pass

        # evaluate only invalid individuals
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_inds:
            ind.fitness.values = toolbox.evaluate(ind)  # type: ignore

        # replace population
        pop[:] = offspring

        # Calculate statistics
        fitnesses = [ind.fitness.values[0] for ind in pop]
        best = tools.selBest(pop, 1)[0]
        best_str = "".join(best)
        best_fit = best.fitness.values[0]
        avg_fit = mean(fitnesses)
        std_fit = stdev(fitnesses) if len(fitnesses) > 1 else 0

        # Calculate diversity (unique individuals in population)
        unique_individuals = len(set("".join(ind) for ind in pop))
        diversity = unique_individuals / population_size

        if track_stats:
            stats_history["generation"].append(gen)
            stats_history["best_fitness"].append(best_fit)
            stats_history["avg_fitness"].append(avg_fit)
            stats_history["std_fitness"].append(std_fit)
            stats_history["diversity"].append(diversity)
            stats_history["best_string"].append(best_str)

        if verbose and (gen % 50 == 0 or best_fit == max_fitness):
            print(
                f"Gen {gen:4d} | Best: {int(best_fit)}/{max_fitness} | Avg: {avg_fit:.2f} | Diversity: {diversity:.2%} | '{best_str}'"
            )

        if best_fit == max_fitness and generations_to_target is None:
            generations_to_target = gen
            if verbose:
                print("-" * 80)
                print(f"\tTarget reached in {gen} generations!")
                print(f"Final string: '{best_str}'")
            break

    # finished max generations or found target
    if generations_to_target is None:
        generations_to_target = generations
        best = tools.selBest(pop, 1)[0]
        best_str = "".join(best)
        if verbose:
            print("-" * 80)
            print(f"Maximum generations ({generations}) reached.")
            print(
                f"Best found: '{best_str}' (Fitness: {int(best.fitness.values[0])}/{max_fitness})"
            )

    return {
        "generations_to_target": generations_to_target,
        "stats_history": stats_history,
        "final_best_fitness": best_fit,
        "final_avg_fitness": avg_fit,
        "final_diversity": diversity,
        "target_reached": best_fit == max_fitness,
        "config": {
            "target": target,
            "population_size": population_size,
            "mutation_method": mutation_method,
            "mutation_rate": mutation_rate,
            "crossover_method": crossover_method,
            "crossover_rate": crossover_rate,
            "selection_method": selection_method,
        },
    }


# ---------- Visualization and Analysis ----------
def plot_evolution_stats(results, title="Evolution Progress"):
    """Plot evolution statistics from a single run."""
    stats = results["stats_history"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    # Best and Average Fitness
    axes[0, 0].plot(
        stats["generation"],
        stats["best_fitness"],
        label="Best Fitness",
        color="green",
        linewidth=2,
    )
    axes[0, 0].plot(
        stats["generation"],
        stats["avg_fitness"],
        label="Avg Fitness",
        color="blue",
        alpha=0.7,
    )
    axes[0, 0].set_xlabel("Generation")
    axes[0, 0].set_ylabel("Fitness")
    axes[0, 0].set_title("Fitness Over Time")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Diversity
    axes[0, 1].plot(
        stats["generation"], stats["diversity"], color="purple", linewidth=2
    )
    axes[0, 1].set_xlabel("Generation")
    axes[0, 1].set_ylabel("Diversity (% unique)")
    axes[0, 1].set_title("Population Diversity")
    axes[0, 1].grid(True, alpha=0.3)

    # Standard Deviation
    axes[1, 0].plot(
        stats["generation"], stats["std_fitness"], color="orange", linewidth=2
    )
    axes[1, 0].set_xlabel("Generation")
    axes[1, 0].set_ylabel("Standard Deviation")
    axes[1, 0].set_title("Fitness Standard Deviation")
    axes[1, 0].grid(True, alpha=0.3)

    # Configuration info
    config = results["config"]
    info_text = f"""
    Target: {config["target"]}
    Population Size: {config["population_size"]}
    Mutation: {config["mutation_method"]} ({config["mutation_rate"]})
    Crossover: {config["crossover_method"]} ({config["crossover_rate"]})
    Selection: {config["selection_method"]}
    
    Generations to target: {results["generations_to_target"]}
    Target reached: {results["target_reached"]}
    """
    axes[1, 1].text(
        0.1,
        0.5,
        info_text,
        fontsize=10,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def compare_configurations(
    results_list, metric="generations_to_target", title="Configuration Comparison"
):
    """Compare multiple GA configurations."""
    labels = []
    values = []

    for result in results_list:
        config = result["config"]
        label = f"{config['mutation_method'][:3]}-{config['crossover_method'][:3]}-{config['selection_method'][:3]}"
        labels.append(label)

        if metric == "generations_to_target":
            values.append(result["generations_to_target"])
        elif metric == "final_diversity":
            values.append(result["final_diversity"])
        elif metric == "final_avg_fitness":
            values.append(result["final_avg_fitness"])

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="steelblue", alpha=0.7)
    plt.xlabel("Configuration (Mutation-Crossover-Selection)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return plt.gcf()


def save_results_to_csv(results, filename="results.csv"):
    """Save experiment results to CSV file."""
    stats = results["stats_history"]

    filepath = DATA_DIR / filename

    with open(filepath, "w", newline="") as csvfile:
        fieldnames = [
            "generation",
            "best_fitness",
            "avg_fitness",
            "std_fitness",
            "diversity",
            "best_string",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(stats["generation"])):
            writer.writerow(
                {
                    "generation": stats["generation"][i],
                    "best_fitness": stats["best_fitness"][i],
                    "avg_fitness": stats["avg_fitness"][i],
                    "std_fitness": stats["std_fitness"][i],
                    "diversity": stats["diversity"][i],
                    "best_string": stats["best_string"][i],
                }
            )

    print(f"Results saved to {filepath}")


def save_summary_to_json(results_list, filename="experiment_summary.json"):
    """Save experiment summary to JSON file."""
    summary = []

    filepath = RESULTS_DIR / filename

    for result in results_list:
        summary.append(
            {
                "config": result["config"],
                "generations_to_target": result["generations_to_target"],
                "target_reached": result["target_reached"],
                "final_best_fitness": result["final_best_fitness"],
                "final_avg_fitness": result["final_avg_fitness"],
                "final_diversity": result["final_diversity"],
            }
        )

    with open(filepath, "w") as jsonfile:
        json.dump(summary, jsonfile, indent=2)

    print(f"Summary saved to {filepath}")


def run_comprehensive_experiment(target="HELLO WORLD", runs_per_config=5):
    """
    Run comprehensive experiments comparing all mutation, crossover, and selection methods.
    """
    mutation_methods = ["simple", "gaussian", "swap", "adaptive"]
    crossover_methods = ["one_point", "two_point", "uniform", "blend"]
    selection_methods = ["tournament", "roulette", "best"]

    all_results = []

    print("=" * 80)
    print("COMPREHENSIVE GENETIC ALGORITHM EXPERIMENT")
    print("=" * 80)
    print(f"Target: '{target}'")
    print(f"Runs per configuration: {runs_per_config}")
    print(
        f"Total configurations: {len(mutation_methods) * len(crossover_methods) * len(selection_methods)}"
    )
    print("=" * 80)

    config_num = 0
    total_configs = (
        len(mutation_methods) * len(crossover_methods) * len(selection_methods)
    )

    for mutation in mutation_methods:
        for crossover in crossover_methods:
            for selection in selection_methods:
                config_num += 1
                print(
                    f"\n[{config_num}/{total_configs}] Testing: {mutation} mutation + {crossover} crossover + {selection} selection"
                )
                print("-" * 80)

                config_results = []
                for run in range(runs_per_config):
                    print(f"  Run {run + 1}/{runs_per_config}...", end=" ")
                    result = run_evolution(
                        target=target,
                        population_size=120,
                        generations=2000,
                        mutation_rate=0.05,
                        crossover_rate=0.7,
                        mutation_method=mutation,
                        crossover_method=crossover,
                        selection_method=selection,
                        track_stats=True,
                        verbose=False,
                    )
                    config_results.append(result)
                    print(f"Done in {result['generations_to_target']} generations")

                # Calculate average performance for this configuration
                avg_generations = mean(
                    [r["generations_to_target"] for r in config_results]
                )
                success_rate = (
                    sum(1 for r in config_results if r["target_reached"])
                    / runs_per_config
                )

                print(f"  Average generations: {avg_generations:.1f}")
                print(f"  Success rate: {success_rate:.1%}")

                # Store the best run for this configuration
                best_run = min(config_results, key=lambda r: r["generations_to_target"])
                best_run["avg_generations"] = avg_generations
                best_run["success_rate"] = success_rate
                all_results.append(best_run)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    # Choose mode: 'demo', 'experiment', or 'single'
    mode = sys.argv[1] if len(sys.argv) > 1 else "experiment"

    if mode == "demo":
        # Quick demonstration with three different configurations
        print("\n" + "=" * 80)
        print("DEMONSTRATION MODE - Running 3 example configurations")
        print("=" * 80)

        # Example 1: Simple mutation + one-point crossover
        result1 = run_evolution(
            target="HELLO WORLD",
            population_size=120,
            generations=2000,
            mutation_rate=0.05,
            crossover_rate=0.7,
            mutation_method="simple",
            crossover_method="one_point",
            selection_method="tournament",
            track_stats=True,
            verbose=True,
        )

        # Example 2: Gaussian mutation + uniform crossover
        print("\n\n")
        result2 = run_evolution(
            target="HELLO WORLD",
            population_size=120,
            generations=2000,
            mutation_rate=0.05,
            crossover_rate=0.7,
            mutation_method="gaussian",
            crossover_method="uniform",
            selection_method="roulette",
            track_stats=True,
            verbose=True,
        )

        # Example 3: Adaptive mutation + two-point crossover
        print("\n\n")
        result3 = run_evolution(
            target="HELLO WORLD",
            population_size=120,
            generations=2000,
            mutation_rate=0.05,
            crossover_rate=0.7,
            mutation_method="adaptive",
            crossover_method="two_point",
            selection_method="tournament",
            track_stats=True,
            verbose=True,
        )

        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        try:
            fig1 = plot_evolution_stats(
                result1, "Configuration 1: Simple + One-Point + Tournament"
            )
            fig1.savefig(
                PLOTS_DIR / "evolution_config1.png", dpi=150, bbox_inches="tight"
            )
            print(f"\tSaved {PLOTS_DIR / 'evolution_config1.png'}")

            fig2 = plot_evolution_stats(
                result2, "Configuration 2: Gaussian + Uniform + Roulette"
            )
            fig2.savefig(
                PLOTS_DIR / "evolution_config2.png", dpi=150, bbox_inches="tight"
            )
            print(f"\tSaved {PLOTS_DIR / 'evolution_config2.png'}")

            fig3 = plot_evolution_stats(
                result3, "Configuration 3: Adaptive + Two-Point + Tournament"
            )
            fig3.savefig(
                PLOTS_DIR / "evolution_config3.png", dpi=150, bbox_inches="tight"
            )
            print(f"\tSaved {PLOTS_DIR / 'evolution_config3.png'}")

            # Comparison chart
            fig_compare = compare_configurations(
                [result1, result2, result3],
                metric="generations_to_target",
                title="Comparison: Generations to Target",
            )
            fig_compare.savefig(
                PLOTS_DIR / "comparison.png", dpi=150, bbox_inches="tight"
            )
            print(f"\tSaved {PLOTS_DIR / 'comparison.png'}")

        except Exception as e:
            print(f"\tCould not create plots: {e}")

        # Save data
        save_results_to_csv(result1, "results_config1.csv")
        save_results_to_csv(result2, "results_config2.csv")
        save_results_to_csv(result3, "results_config3.csv")
        save_summary_to_json([result1, result2, result3], "demo_summary.json")

    elif mode == "experiment":
        # Full comprehensive experiment
        target = sys.argv[2] if len(sys.argv) > 2 else "HELLO WORLD"
        runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        top_n = int(sys.argv[4]) if len(sys.argv) > 4 else 2

        all_results = run_comprehensive_experiment(target=target, runs_per_config=runs)

        # Save comprehensive results
        save_summary_to_json(all_results, "comprehensive_experiment.json")

        # Sort by performance
        sorted_results = sorted(all_results, key=lambda r: r["avg_generations"])

        # Create comparison visualizations
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)

        try:
            # Top N best configurations comparison
            top_results = sorted_results[: min(top_n, 10)]  # max 10 for readability

            fig = compare_configurations(
                top_results,
                metric="generations_to_target",
                title=f"Top {len(top_results)} Configurations - Target: '{target}'",
            )
            fig.savefig(
                PLOTS_DIR / f"top_{len(top_results)}_configurations.png",
                dpi=150,
                bbox_inches="tight",
            )
            print(f"\tSaved {PLOTS_DIR / f'top_{len(top_results)}_configurations.png'}")

            # Create detailed plots for each top configuration
            for i, result in enumerate(top_results, 1):
                config = result["config"]
                title = f"Rank #{i}: {config['mutation_method'].capitalize()} + {config['crossover_method'].capitalize()} + {config['selection_method'].capitalize()}"

                fig_detail = plot_evolution_stats(result, title)
                filename = f"rank_{i}_config.png"
                fig_detail.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
                print(f"\tSaved {PLOTS_DIR / filename}")

                # Save detailed CSV for each top configuration
                csv_filename = f"rank_{i}_config.csv"
                save_results_to_csv(result, csv_filename)

            # Print summary of top configurations
            print("\n" + "=" * 80)
            print(f"TOP {len(top_results)} CONFIGURATIONS SUMMARY")
            print("=" * 80)

            for i, result in enumerate(top_results, 1):
                config = result["config"]
                print(f"\n#{i} CONFIGURATION:")
                print(f"\tMutation: {config['mutation_method']}")
                print(f"\tCrossover: {config['crossover_method']}")
                print(f"\tSelection: {config['selection_method']}")
                print(f"\tAverage generations: {result['avg_generations']:.1f}")
                print(f"\tSuccess rate: {result['success_rate']:.1%}")
                print(f"\tBest run: {result['generations_to_target']} generations")

            print("\n" + "=" * 80)

        except Exception as e:
            print(f"\tCould not create plots: {e}")
            import traceback

            traceback.print_exc()

    elif mode == "single":
        # Run a single custom configuration
        target = sys.argv[2] if len(sys.argv) > 2 else "HELLO WORLD"
        mutation = sys.argv[3] if len(sys.argv) > 3 else "adaptive"
        crossover = sys.argv[4] if len(sys.argv) > 4 else "uniform"
        selection = sys.argv[5] if len(sys.argv) > 5 else "tournament"

        result = run_evolution(
            target=target,
            population_size=120,
            generations=2000,
            mutation_rate=0.05,
            crossover_rate=0.7,
            mutation_method=mutation,
            crossover_method=crossover,
            selection_method=selection,
            track_stats=True,
            verbose=True,
        )

        try:
            fig = plot_evolution_stats(
                result, f"Custom Configuration: {mutation}-{crossover}-{selection}"
            )
            fig.savefig(PLOTS_DIR / "custom_run.png", dpi=150, bbox_inches="tight")
            print(f"\n\tSaved {PLOTS_DIR / 'custom_run.png'}")
        except Exception as e:
            print(f"\n\tCould not create plot: {e}")

        save_results_to_csv(result, "custom_run.csv")
        save_summary_to_json([result], "custom_run.json")

    else:
        print("Usage:")
        print(
            "\tpython main.py                         # Run comprehensive experiment (default)"
        )
        print(
            "\tpython main.py experiment [target] [runs] [top_n]  # Run comprehensive experiment with custom params"
        )
        print(
            "\tpython main.py demo                    # Run quick demonstration with 3 configurations"
        )
        print(
            "\tpython main.py single [target] [mutation] [crossover] [selection]  # Run single custom configuration"
        )
        print("\nMutation methods: simple, gaussian, swap, adaptive")
        print("Crossover methods: one_point, two_point, uniform, blend")
        print("Selection methods: tournament, roulette, best")
        print("\nExamples:")
        print(
            "\tpython main.py                         # Default: comprehensive with 2 runs, top 2 configs"
        )
        print(
            "\tpython main.py experiment 'HELLO' 3 5  # Target='HELLO', 3 runs per config, show top 5"
        )
        print("\tpython main.py single 'TEST' adaptive uniform tournament")
