package mqo_chimera.solver;

import java.io.File;
import java.io.PrintWriter;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.solver.cplex.LinearSolver;
import mqo_chimera.util.AmesUtil;

/**
 * Uses previously generated test cases to benchmark the linear solver based on CPLEX.
 * 
 * @author immanueltrummer
 *
 */
public class BenchmarkLinear {
	public static void main(String[] args) throws Exception {
		// Initialize description of Ames machine
		AmesUtil.initAmes();
		// Generate linear solver
		LinearSolver linearSolver = new LinearSolver();
		// Iterate over test classes
		for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
			// Generate path to file storing summary information
			String summaryFilePath = "mqo/linear/" + testClass.configurationID() + "_summary";
			// Open summary file and write header
			File summaryFile = new File(summaryFilePath);
			PrintWriter summaryWriter = new PrintWriter(summaryFile);
			summaryWriter.println("testcaseCtr,solverMillis,optimalCost");
			// Iterate over test cases
			for (int testcaseCtr=0; testcaseCtr<BenchmarkConfiguration.nrTestcases; ++testcaseCtr) {
				// Generate path to test case
				String testcasePath = "mqo/testcases/" + testClass.configurationID() + "T" + testcaseCtr;
				// Read test case from disc
				ChimeraMqoMapping mapping = ChimeraMqoMapping.readMapping(testcasePath);
				// Solve test case
				//linearSolver.solve(mapping.problem);
				linearSolver.solveChimeraQubo(mapping);
				// Generate path to quality curve
				String solutionPath = "mqo/qubolinear/" + testClass.configurationID() + "T" + testcaseCtr + "_costCurve";
				// Write out quality curve
				linearSolver.writeLastCurve(solutionPath);
				// Write out performance information
				summaryWriter.println(testcaseCtr + "," + linearSolver.lastRunSolverMillis + "," + linearSolver.lastRunOptimalCost);
				// Output progress reporting
				System.out.println("Solved test case " + testcaseCtr + " of test class " + testClass);
			}
			// close performance file
			summaryWriter.close();			
		}
	}
}
