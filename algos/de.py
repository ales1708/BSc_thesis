import numpy as np

# from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init

from utils.Algorithm import Algorithm, Individual, default_individual_init
from niapy.util.array import objects_to_array

def cross_rand1(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses three different random individuals from population to perform mutation.

    Mutation:
        Name: DE/rand/1

        :math:`\mathbf{x}_{r_1, G} + differential_weight \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}`
        where :math:`r_1, r_2, r_3` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: Mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
    r = rng.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
    x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)

class DifferentialEvolution(Algorithm):
    r"""Implementation of Differential evolution algorithm.

    Algorithm:
         Differential evolution algorithm

    Date:
        2018

    Author:
        Uros Mlakar and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.

    Attributes:
        Name (List[str]): List of string of names for algorithm.
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.
        strategy (Callable[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, Dict[str, Any]]): crossover and mutation strategy.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['DifferentialEvolution', 'DE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359."""

    def __init__(self, population_size=50, differential_weight=1, crossover_probability=0.8, strategy=cross_rand1,
                 *args, **kwargs):
        """Initialize DifferentialEvolution.

        Args:
            population_size (Optional[int]): Population size.
            differential_weight (Optional[float]): Differential weight (differential_weight).
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, list], numpy.ndarray]]):
                Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size,
                         initialization_function=kwargs.pop('initialization_function', default_individual_init),
                         individual_type=kwargs.pop('individual_type', Individual), *args, **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def set_parameters(self, population_size=50, differential_weight=1, crossover_probability=0.8, strategy=cross_rand1,
                       **kwargs):
        r"""Set the algorithm parameters.

        Args:
            population_size (Optional[int]): Population size.
            differential_weight (Optional[float]): Differential weight (differential_weight).
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, list], numpy.ndarray]]):
                Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size,
                               initialization_function=kwargs.pop('initialization_function', default_individual_init),
                               individual_type=kwargs.pop('individual_type', Individual), **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def get_parameters(self):
        r"""Get parameters values of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'differential_weight': self.differential_weight,
            'crossover_probability': self.crossover_probability,
            'strategy': self.strategy
        })
        return d

    def evolve(self, pop, xb, task, **kwargs):
        r"""Evolve population.

        Args:
            pop (numpy.ndarray): Current population.
            xb (numpy.ndarray): Current best individual.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New evolved populations.

        """
        return objects_to_array(
            [self.individual_type(x=self.strategy(pop, i, self.differential_weight, self.crossover_probability, self.rng, x_b=xb), task=task, rng=self.rng, e=True) for i
             in range(len(pop))])

    def selection(self, population, new_population, best_x, best_fitness, task, **kwargs):
        r"""Operator for selection.

        Args:
            population (numpy.ndarray): Current population.
            new_population (numpy.ndarray): New Population.
            best_x (numpy.ndarray): Current global best solution.
            best_fitness (float): Current global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New selected individuals.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        """
        arr = objects_to_array([e if e.f < population[i].f else population[i] for i, e in enumerate(new_population)])
        best_x, best_fitness = self.get_best(arr, np.asarray([e.f for e in arr]), best_x, best_fitness)
        return arr, best_x, best_fitness

    def post_selection(self, pop, task, xb, fxb, **kwargs):
        r"""Apply additional operation after selection.

        Args:
            pop (numpy.ndarray): Current population.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New population.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        """
        return pop, xb, fxb

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Differential Evolution algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Current best individual.
            best_fitness (float): Current best individual function/fitness value.
            **params (dict): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.evolve`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.selection`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.post_selection`

        """
        new_population = self.evolve(population, best_x, task)
        population, best_x, best_fitness = self.selection(population, new_population, best_x, best_fitness, task=task)
        population, best_x, best_fitness = self.post_selection(population, task, best_x, best_fitness)
        population_fitness = np.asarray([x.f for x in population])
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}
