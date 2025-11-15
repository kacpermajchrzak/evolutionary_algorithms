import random
import string
from pathlib import Path
from statistics import mean, stdev
from deap import base, creator, tools
import matplotlib.pyplot as plt

# ---------- Directory Setup ----------
BASE_DIR = Path(__file__).parent
PLOTS_DIR = BASE_DIR / "plots"

PLOTS_DIR.mkdir(exist_ok=True)

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
    else:
        toolbox.register("mate", cx_one_point)

    # mutation registration (for adaptive we will re-register inside loop)
    if mutation_method == "simple":
        toolbox.register("mutate", mutate_simple, mutation_rate=mutation_rate)
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


# ---------- Example usage ----------
if __name__ == "__main__":
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

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    try:
        fig1 = plot_evolution_stats(
            result1, "Configuration 1: Simple + One-Point + Tournament"
        )
        fig1.savefig(PLOTS_DIR / "evolution_config1.png", dpi=150, bbox_inches="tight")
        print(f"\tSaved {PLOTS_DIR / 'evolution_config1.png'}")

    except Exception as e:
        print(f"\tCould not create plots: {e}")
