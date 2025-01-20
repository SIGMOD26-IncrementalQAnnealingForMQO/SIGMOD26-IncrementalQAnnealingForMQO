package mqo_chimera.testcases;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.util.AmesUtil;

/**
 * Applies some additional consistency tests to the test cases.
 * 
 * @author immanueltrummer
 *
 */
public class ConsistencyTests {
	public static void main(String[] args) throws Exception {
		// Initialize description of Ames machine
		AmesUtil.initAmes();
		// Iterate over test classes
		for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
			// Iterate over test cases
			for (int testcaseCtr=0; testcaseCtr<BenchmarkConfiguration.nrTestcases; ++testcaseCtr) {
				// Generate path to test case
				String configurationID = testClass.configurationID();
				String testcasePath = "mqo/testcases/" + configurationID + "T" + testcaseCtr;
				// Read test case from disc
				ChimeraMqoMapping mapping = ChimeraMqoMapping.readMapping(testcasePath);
				ChimeraMqoProblem problem = mapping.problem;
				boolean allowIndependentProcessing = problem.allowIndependentProcessing;
				// Get optimal execution cost
				double optimalCost = problem.executionCost(problem.plantedPlanSelections);
				// Make sure that randomly selected plan selections have no lower than optimal cost
				for (int i=0; i<100; ++i) {
					int[] solution = ChimeraFactory.pickRandomPlans(problem, allowIndependentProcessing);
					double cost = problem.executionCost(solution);
					assert(optimalCost <= cost);
					if (optimalCost > cost) {
						System.out.println("ERROR: Randomly generated solution is better than optimal!");
					}
				}
				// Notify on progress
				System.out.println("Testcase " + testcaseCtr + " passed consistency tests!");
			}			
		}
	}
}
