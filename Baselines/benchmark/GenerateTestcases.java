package mqo_chimera.benchmark;

import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.util.AmesUtil;

public class GenerateTestcases {
	public static void main(String[] args) throws Exception {
		// Initialize description of Ames machine
		AmesUtil.initAmes();
		// Iterate over test classes
		for (TestcaseClass testClass : BenchmarkConfiguration.testcaseClasses) {
			// Iterate over test cases
			for (int testcaseCtr=0; testcaseCtr<BenchmarkConfiguration.nrTestcases; ++testcaseCtr) {
				// Generate test case
				ChimeraMqoMapping mapping = ChimeraFactory.produceStandardTestcase(testClass);
				// Generate random gauge transformations
				mapping.generateGaugeTransformations();
				// Generate path to new test case
				String testcasePath = "mqo/testcases/" + testClass.configurationID() + "T" + testcaseCtr;
				// Write test case to disc
				mapping.toFile(testcasePath);
				// Generate progress report
				System.out.println("Generated test case " + testcaseCtr);
			}
			// Generate progress report
			System.out.println("Generated all test cases for class " + testClass.toString());
		}
	}
}
