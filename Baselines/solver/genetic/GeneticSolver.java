package mqo_chimera.solver.genetic;

import java.util.Arrays;

import org.jgap.Chromosome;
import org.jgap.Configuration;
import org.jgap.Gene;
import org.jgap.Genotype;
import org.jgap.IChromosome;
import org.jgap.impl.DefaultConfiguration;
import org.jgap.impl.IntegerGene;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.solver.Solver;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.MqoSolution;

/**
 * A genetic algorithm that solves MQO problems.
 * 
 * @author immanueltrummer
 *
 */
public class GeneticSolver extends Solver {
	/**
	 * The population size.
	 */
	final int populationSize;

	public GeneticSolver(int populationSize) {
		this.populationSize = populationSize;
	}
	/**
	 * Represents a MQO problem instance as chromosomes and simulates an evolution
	 * until the timeout.
	 */
	@Override
	public MqoSolution solve(ChimeraMqoProblem problem) throws Exception {
		// Extract MQO problem instance and problem dimensions
		int nrQueries = problem.nrQueries;
		int nrPlans = problem.nrPlansPerQuery;
		// Configure genetic algorithm
		Configuration.reset();
		Configuration gaConfiguration = new DefaultConfiguration();
		gaConfiguration.setPreservFittestIndividual(true);
		// Set fitness function
		MqoFitnessFunction fitnessFunction = new MqoFitnessFunction(problem);
		gaConfiguration.setFitnessFunction(fitnessFunction);
		// Set sample chromosome
		Gene[] sampleGenes = new Gene[nrQueries];
		for (int query=0; query<nrQueries; ++query) {
			sampleGenes[query] = new IntegerGene(gaConfiguration, 0, nrPlans - 1);
		}
		IChromosome sampleChromosome = new Chromosome(gaConfiguration, sampleGenes);
		gaConfiguration.setSampleChromosome(sampleChromosome);
		// Set population size
		gaConfiguration.setPopulationSize(populationSize);
		// Generate population
		Genotype population = Genotype.randomInitialGenotype(gaConfiguration);
		// Initialize benchmark variables
		Arrays.fill(lastRunCheckpointCost, Double.POSITIVE_INFINITY);
		long startMillis = System.currentTimeMillis();
		long elapsedMillis = 0;
		// Evolve
		do {
			population.evolve();
			IChromosome bestChromosome = population.getFittestChromosome();
			int[] bestSelections = MqoFitnessFunction.extractPlanSelections(bestChromosome);
			double bestCost = problem.executionCost(bestSelections);
			elapsedMillis = System.currentTimeMillis() - startMillis;
			updateStats(bestCost, elapsedMillis);
		} while (elapsedMillis < BenchmarkConfiguration.timeoutMillis);
		// Return best solution
		IChromosome bestChromosome = population.getFittestChromosome();
		int[] bestSelections = MqoFitnessFunction.extractPlanSelections(bestChromosome);
		return new MqoSolution(problem, bestSelections);
	}
	@Override
	public String solverID() {
		return "GEN" + populationSize;
	}
}
