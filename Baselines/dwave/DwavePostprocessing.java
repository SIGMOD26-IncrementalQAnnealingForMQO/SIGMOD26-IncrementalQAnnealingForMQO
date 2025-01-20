package mqo_chimera.dwave;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.testcases.QuadraticMqoSolution;
import mqo_chimera.util.AmesUtil;

/**
 * Reads solutions produced by D-Wave, calculates the energy level, and compares with the
 * optimal (=minimal) energy level.
 * 
 * @author immanueltrummer
 *
 */
public class DwavePostprocessing {
	/**
	 * Reads a solution produced by D-Wave to the Ising problem representing the MQO problem.
	 * 
	 * @param solutionPath	path to the file containing the solution
	 * @return				Boolean vector assigning each qubit/spin to a Boolean value
	 * @throws Exception 
	 */
	public static boolean[][] readSolutions(String solutionPath) throws Exception {
		// Dimension variables
		int nrSamples = BenchmarkConfiguration.nrSamplesPerTransformation;
		int nrQubits = 1152;
		// This will contain the result - first index is the sample, the second the qubit
		boolean[][] qubitValues = new boolean[nrSamples][nrQubits];
		// Open solution file
		FileReader fileReader = new FileReader(solutionPath);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		// Iterate over the number of samples per gauge transformation
		for (int sampleCtr=0; sampleCtr<nrSamples; ++sampleCtr) {
			// Skip introductory lines
			bufferedReader.readLine();	// Line stating solution energy
			bufferedReader.readLine();	// Line stating solution length
			bufferedReader.readLine();	// Line stating number of occurrences
			bufferedReader.readLine();	// header
			// Read qubit values one after the other
			for (int qubitIndex=0; qubitIndex<nrQubits; ++qubitIndex) {
				// Read line representing assignment for one qubit
				String line = bufferedReader.readLine();
				// Values are separated by comma
				String[] splits = line.split(",");
				// Read values
				Integer qubitIndexRead = Integer.parseInt(splits[0]);
				Integer qubitValueRead = Integer.parseInt(splits[1]);
				// Make sure that read qubit value corresponds to expected one
				assert(qubitIndexRead.equals(qubitIndex));
				// Transform integer value into Boolean value
				qubitValues[sampleCtr][qubitIndex] = qubitValueRead.equals(1) ? true : false;
			}			
		}
		// Close solution file
		bufferedReader.close();
		// Return solution
		return qubitValues;
	}
	/**
	 * Given a solution to a transformed problem and the transformation, transform the solution
	 * back such that it refers to the original problem afterwards.
	 * 
	 * @param solutionToTransformed	solution to transformed problem
	 * @param transformation		gauge transformation to which the solution refers 
	 * @return						solution to non-transformed problem
	 */
	public static boolean[] transformBack(boolean[] solutionToTransformed, boolean[] transformation) {
		// Obtain number of spins
		int nrSpins = solutionToTransformed.length;
		assert(nrSpins == transformation.length);
		// Will contain result of this function
		boolean[] originalSolution = new boolean[nrSpins];
		// Iterate over solution spins and inverse them depending on the transformation
		for (int spin=0; spin<nrSpins; ++spin) {
			originalSolution[spin] = transformation[spin] ? 
					solutionToTransformed[spin] : !solutionToTransformed[spin];
		}
		return originalSolution;
	}

	public static void main(String[] args) throws Exception {
		// Determines which fraction of cost samples to log
		final int logEveryIth = 10;
		// Initialize description of Ames machine
		AmesUtil.initAmes();
		// Iterate over test case classes
		for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
			// Generate path to file storing solution quality information
			String configurationID = testClass.configurationID();
			// Iterate over test cases
			for (int testcaseCtr=0; testcaseCtr<BenchmarkConfiguration.nrTestcases; ++testcaseCtr) {
				// Generate path to test case
				String testcasePath = "mqo/testcases/" + configurationID + "T" + testcaseCtr;
				// Read test case from disc
				ChimeraMqoMapping quboMapping = ChimeraMqoMapping.readMapping(testcasePath);
				// Transform into Ising mapping
				ChimeraMqoMapping isingMapping = quboMapping.toIsing();
				// Generate path to curve file
				String costCurvePath = "mqo/dwave/" + configurationID + "_T" + testcaseCtr + "_costCurve";
				// Open curve file
				PrintWriter curveWriter = new PrintWriter(costCurvePath);
				// Write curve header
				curveWriter.println("elapsedMillis,bestCost");
				// Initialize best cost value
				double bestCost = Double.POSITIVE_INFINITY;
				// Counts the total number of samples over all gauge transformations
				int sampleCtr = 1;
				// Process solutions obtained for different gauge transformations
				for (int transformationCtr=0; transformationCtr<BenchmarkConfiguration.nrTransformations; ++transformationCtr) {
					// Generate path to solution file generated by D-Wave solver
					String dwaveSolutionPath = "mqo/dwave/" + configurationID + "_ising" + 
							"_T" + testcaseCtr + "_G" + transformationCtr + "_solutions";
					// Read solutions to transformed problem from disc
					boolean[][] transformedDwaveQubitValues = readSolutions(dwaveSolutionPath);
					// Calculate execution cost associated with each solution
					for (int gaugeSampleCtr=0; gaugeSampleCtr<BenchmarkConfiguration.nrSamplesPerTransformation; 
							++gaugeSampleCtr) {
						// Obtain gauge transformation to which the solution refers
						boolean[] gaugeTransformation = isingMapping.gaugeTransformations[transformationCtr];
						// Transform solution back to solution for original problem
						boolean[] dwaveQubitValues = transformBack(transformedDwaveQubitValues[gaugeSampleCtr], gaugeTransformation);
						// Calculate plan selections
						int[] planSelections = new QuadraticMqoSolution(quboMapping, dwaveQubitValues).planSelections;
						// Calculate execution cost
						double executionCost = isingMapping.problem.executionCost(planSelections);
						// Calculate minimal execution cost
						bestCost = Math.min(bestCost, executionCost);
						// Calculate elapsed number of milliseconds - take into account annealing time and read out time
						double elapsedMillis = 0.247 * 2 * sampleCtr;
						// Write execution cost to file
						if ((gaugeSampleCtr % logEveryIth) == 0) {
							curveWriter.println(elapsedMillis + "," + bestCost);							
						}
						// Advance total sample counter
						++sampleCtr;
					}
					// notify user of progress
					System.out.println("Processed transformation " + transformationCtr);
				}
				curveWriter.close();
				// notify user of progress
				System.out.println("Processed test case nr. " + testcaseCtr);
			}
			// Progress output
			System.out.println("Processed test case class " + testClass);
		}
	}

}
