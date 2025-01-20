package mqo_chimera.solver.genetic;

import java.util.Map.Entry;
import mqo_chimera.testcases.ChimeraMqoProblem;
import mqo_chimera.testcases.PlanCoupling;
import org.jgap.FitnessFunction;
import org.jgap.IChromosome;

/**
 * Evaluates the fitness of a MQO solution.
 * 
 * @author immanueltrummer
 *
 */
public class MqoFitnessFunction extends FitnessFunction {
	/**
	 * Used to verify class version.
	 */
	private static final long serialVersionUID = 1L;
	/**
	 * The MQO problem instance to which the solutions refer.
	 */
	final ChimeraMqoProblem problem;
	/**
	 * Number of queries for which to select plans.
	 */
	final int nrQueries;
	/**
	 * Number of plans to consider per query.
	 */
	final int nrPlans;
	/**
	 * An upper bound on the cost of each solution.
	 */
	final double upperCostBound;
	
	public MqoFitnessFunction(ChimeraMqoProblem problem) {
		this.problem = problem;
		this.nrQueries = problem.nrQueries;
		this.nrPlans = problem.nrPlansPerQuery;
		this.upperCostBound = calculateUpperCostBound(problem);
	}
	/**
	 * Calculates an upper bound on the cost of each possible solution to the
	 * given MQO problem instance by summing up maximal plan costs over all
	 * queries. Does not take into account interactions between different plans
	 * as we assume all those actions to be benign.
	 * 
	 * @param problem	an MQO problem instance
	 * @return			a positive upper cost bound
	 */
	public double calculateUpperCostBound(ChimeraMqoProblem problem) {
		//  Verify that all plan interactions are benign
		for (Entry<PlanCoupling, Double> entry : problem.interactions.entrySet()) {
			double costDelta = entry.getValue();
			assert(costDelta <= 0);
		}
		// Calculate an upper cost bound
		double upperBound = 0;
		for (int query=0; query<nrQueries; ++query) {
			double maxPlanCost = 0;
			for (int plan=0; plan<nrPlans; ++plan) {
				double planCost = problem.planExecutionCost(query, plan);
				maxPlanCost = Math.max(maxPlanCost, planCost);
			}
			upperBound += maxPlanCost;
		}
		return upperBound;
	}
	/**
	 * Decodes a chromosome by extracting the selected plan for each query.
	 * 
	 * @param solution	a chromosome representing a solution to a MQO problem
	 * @return			a vector of integers representing plan selections
	 */
	public static int[] extractPlanSelections(IChromosome solution) {
		// extract problem dimensions from chromosome
		int nrQueries = solution.size();
		// will contain the result of this function
		int[] planSelections = new int[nrQueries];
		// iterate over queries and extract the selected plan
		for (int query=0; query<nrQueries; ++query) {
			planSelections[query] = (Integer)solution.getGene(query).getAllele();
		}
		return planSelections;
	}
	/**
	 * Decodes the chromosome into plan selections and returns a
	 * fitness value that is based on the execution cost for the
	 * given plan selections.
	 * 
	 * @param solution	chromosome representing plan selections
	 * @return			fitness value (anti-monotone in the execution cost)
	 */
	@Override
	protected double evaluate(IChromosome solution) {
		int[] planSelections = extractPlanSelections(solution);
		double executionCost = problem.executionCost(planSelections);
		double fitness = upperCostBound - executionCost;
		assert(fitness >= 0);
		return fitness;
	}
}
