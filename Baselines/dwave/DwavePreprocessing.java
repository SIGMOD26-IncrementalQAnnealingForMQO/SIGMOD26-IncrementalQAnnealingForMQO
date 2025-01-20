package mqo_chimera.dwave;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

import mqo_chimera.benchmark.BenchmarkConfiguration;
import mqo_chimera.benchmark.TestcaseClass;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.util.AmesUtil;

/**
 * The D-Wave solver is accessed over a program implemented in C - this preprocessing
 * step writes out the testcase descriptions in QUBO format that can be processed
 * by the C program and additionally generates an invocation script for the C program.
 * 
 * @author immanueltrummer
 *
 */
public class DwavePreprocessing {
	/**
	 * Writes out a bash script that is used to invoke a C program that uses
	 * D-Wave solvers to solve the test cases and produces corresponding output
	 * file that are processed by the Postprocessing Java program.
	 * 
	 * @throws Exception
	 */
	static void writeDwaveScript() throws Exception {
		// Generate run script (assuming that the script is invoked from inside the D-Wave directory)
		String scriptPath = "mqo/dwave/run_all.sh";
		// Open file for writing
		PrintWriter writer = new PrintWriter(scriptPath);
		// Bash header
		writer.println("#!/bin/bash");
		// Include dynamic D-Wave library
		writer.println("export DYLD_LIBRARY_PATH=.");
		// Iterate over test case classes
		for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
			// Create path of file for outputting performance results
			String outputPath = testClass.configurationID() + "_performance";
			// Create path of file holding logging output
			String loggingPath = testClass.configurationID() + "_logging";
			// Create common input path prefix
			String inputPath = testClass.configurationID() + "_ising" + "_T";
			// Write invocation command
			writer.println("./dwave_cpart " + 
					BenchmarkConfiguration.nrTestcases + " " +			// number of test cases 
					BenchmarkConfiguration.nrTransformations + " " +	// number of transformations
					inputPath + " "										// input prefix 
					+ "REMOTE" + " " +									// solver/connection 
					"-" + " " +											// whether to embed the problem 
					"DO_SOLVE" + " " +									// whether to solve the problem
					"0" + " " +											// weight for equality constraints (only used if embedding is done by C) 
					outputPath + " " +									// path to performance output file
					"> " + loggingPath									// path to log file
					);			
		}
		// Close script file
		writer.close();
	}
	/**
	 * Reads the optimal cost values for each test case that belongs to
	 * the given test case class.
	 * 
	 * @param testClass		describes the class of test cases
	 * @return				a vector containing the optimal cost values for each test case
	 * @throws Exception
	 */
	static double[] readOptimalCost(TestcaseClass testClass) throws Exception {
		// Generate path to linear solver output
		String summaryFilePath = "mqo/linear/" + testClass.configurationID() + "_summary";
		// This will contain the optimal cost values
		double[] optimalCost = new double[BenchmarkConfiguration.nrTestcases];
		// Open summary file
		FileReader fileReader = new FileReader(summaryFilePath);
		BufferedReader bufferedReader = new BufferedReader(fileReader);
		// Skip header and assert that it's the expected one
		String header = bufferedReader.readLine();
		assert(header.equals("testcaseCtr,solverMillis,optimalCost"));
		// Read line by line
		int testcaseCtr = 0;
		String line;
		while ((line = bufferedReader.readLine()) != null) {
			String[] splits = line.split(",");
			assert(testcaseCtr == Integer.parseInt(splits[0]));
			optimalCost[testcaseCtr] = Double.parseDouble(splits[2]);
			++testcaseCtr;
		}
		// Close summary file
		bufferedReader.close();
		// Return optimal cost values
		return optimalCost;
	}
	
	public static void main(String[] args) throws Exception {
		// Initialize description of Ames machine
		AmesUtil.initAmes();
		// Iterate over test classes
		for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
			// Generate path to file storing performance information
			String configurationID = testClass.configurationID();
			// Iterate over test cases
			for (int testcaseCtr=0; testcaseCtr<BenchmarkConfiguration.nrTestcases; ++testcaseCtr) {
				// Generate path to test case
				String testcasePath = "mqo/testcases/" + configurationID + "T" + testcaseCtr;
				// Read test case from disc
				ChimeraMqoMapping quboMapping = ChimeraMqoMapping.readMapping(testcasePath);
				// Transform QUBO problem into Ising problem
				ChimeraMqoMapping isingMapping = quboMapping.toIsing();
				// Iterate over gauge transformations
				for (int transformationCtr=0; transformationCtr<BenchmarkConfiguration.nrTransformations; ++transformationCtr) {
					// Generate path to Ising file
					String isingPath = "mqo/dwave/" + configurationID + "_ising" + "_T" + testcaseCtr + "_G" + transformationCtr;
					// Write out transformed Ising weights
					isingMapping.transformedWeightsToFile(isingPath, "Ising description - " + configurationID, transformationCtr);
				}
			}
		}
		// Write script to run all test cases on D-Wave
		writeDwaveScript();
	}
}
