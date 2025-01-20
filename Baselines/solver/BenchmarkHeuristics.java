package mqo_chimera.solver;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.solver.climbing.HillClimber;
import mqo_chimera.solver.genetic.GeneticSolver;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.util.AmesUtil;

/**
 * Benchmarks a simple iterative improvement randomized algorithm that randomly
 * generates solutions and improves them via hill climbing. We use the hill
 * climber with a timeout and write out the solution (plan selections) it finds
 * for each test case as well as aggregate information about its performance.
 * 
 * @author immanueltrummer
 *
 */
public class BenchmarkHeuristics {
	/**
	 * Writes a solution to a MQO problem (a plan selection for each query) to a file on disc.
	 * 
	 * @param filePath		path of the file to create
	 * @param solution		an integer vector containing for each query the index of the selected plan
	 * @throws Exception
	 */
	/*
	public static void writePlanSelections(String filePath, int[] solution) throws Exception {
		// Open file for writing
		PrintWriter writer = new PrintWriter(filePath);
		// Write header
		writer.println("Query,plan");
		// Write plan selections
		int nrQueries = solution.length;
		for (int query=0; query<nrQueries; ++query) {
			int selectedPlan = solution[query];
			writer.println(query + "," + selectedPlan);
		}
		// Close file
		writer.close();
	}
	*/
	
	public static void main(String[] args) throws Exception {
		// Initialize description of Ames machine
		AmesUtil.initAmes();
		// Generate heuristic solvers
		//Solver[] solvers = new Solver[] {new HillClimber(), new GeneticSolver(50), new GeneticSolver(200)};
		Solver[] solvers = new Solver[] {new GeneticSolver(50), new GeneticSolver(200)};
		// Iterate over solvers
		for (Solver solver : solvers) {
			// Iterate over test classes
			for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
				// Code warmup
				{
					ChimeraMqoMapping randomMapping = ChimeraFactory.produceStandardTestcase(testClass);
					solver.solve(randomMapping.problem);
				}
				// Get test class and solver ID
				String configurationID = testClass.configurationID();
				String solverID = solver.solverID();
				// Iterate over test cases
				for (int testcaseCtr=0; testcaseCtr<BenchmarkConfiguration.nrTestcases; ++testcaseCtr) {
					// Generate path to test case
					String testcasePath = "mqo/testcases/" + configurationID + "T" + testcaseCtr;
					// Read test case from disc
					ChimeraMqoMapping mapping = ChimeraMqoMapping.readMapping(testcasePath);
					// Solve
					solver.solve(mapping.problem);
					// Generate path to curve file
					String curvePath = "mqo/heuristic/" + configurationID + "_" + 
											solverID + "T" + testcaseCtr + "_costCurve";
					// Write curve file
					solver.writeLastCurve(curvePath);
					// Notify on progress
					System.out.println("Treated test case " + testcaseCtr);
				}
			}			
		}
	}
}
