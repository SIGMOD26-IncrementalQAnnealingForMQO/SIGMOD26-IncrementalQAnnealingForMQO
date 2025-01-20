package mqo_chimera.benchmark;

import java.util.stream.LongStream;

/**
 * Describes which test cases have to be generated and solved.
 * 
 * @author immanueltrummer
 *
 */
public class BenchmarkConfiguration {
	/**
	 * Number of test cases per test case class.
	 */
	public static int nrTestcases = 20;
	/**
	 * The times (expressed as milliseconds since optimization start) at
	 * which we compare solution quality between different solvers.
	 */
	/*public static long[] benchmarkTimes = new long[] {
		1, 10, 100, 1000, 10000, 100000
	};*/
	/*public static long[] benchmarkTimes = LongStream.rangeClosed(0, (long)((60000-500)/500))
			.map(x -> x*500 + 500).toArray();*/

	public static long[] benchmarkTimes = new long[] {
		10000, 20000, 30000, 40000, 50000, 60000
	};

	/*public static long[] benchmarkTimes = new long[] {
		60000
	};*/



	/**
	 * The number of checkpoints at which we compare different solvers.
	 */
	public static final int nrBenchmarkTimes = benchmarkTimes.length;
	/**
	 * The number of milliseconds until a timeout is registered.
	 */
	public static long timeoutMillis = benchmarkTimes[nrBenchmarkTimes - 1];
	/**
	 * Number of gauge transformations to try per test case.
	 */
	public static int nrTransformations = 10;
	/**
	 * Number of samples that D-Wave takes per gauge transformation.
	 */
	public static int nrSamplesPerTransformation = 100;
	/**
	 * The considered test case classes (characterized by a specific number of queries, query plans per query etc.).
	 */
	public static TestcaseClass[] testcaseClasses = new TestcaseClass[] {
		new TestcaseClass(537, 2, false),
		new TestcaseClass(253, 3, false),
		new TestcaseClass(140, 4, false),
		new TestcaseClass(108, 5, false),
		/*
		// each qubit represents the selection between two alternative query plans
		new TestcaseClass(1097, 1, 1000, 6, 1, true),
		new TestcaseClass(1097, 1, 750, 6, 1, true),
		new TestcaseClass(1097, 1, 500, 6, 1, true),
		new TestcaseClass(1097, 1, 250, 6, 1, true),
		// two qubits are required per query
		new TestcaseClass(537, 2, 1000, 6, 1, true),
		new TestcaseClass(537, 2, 750, 6, 1, true),
		new TestcaseClass(537, 2, 500, 6, 1, true),
		new TestcaseClass(537, 2, 250, 6, 1, true),
		
		new TestcaseClass(537, 2, 1000, 6, 1, false),
		new TestcaseClass(537, 2, 750, 6, 1, false),
		new TestcaseClass(537, 2, 500, 6, 1, false),
		new TestcaseClass(537, 2, 250, 6, 1, false),
		// three qubits per query
		new TestcaseClass(253, 3, 500, 6, 1, true),
		new TestcaseClass(253, 3, 250, 6, 1, true),		
		new TestcaseClass(253, 3, 500, 6, 1, false),
		new TestcaseClass(253, 3, 250, 6, 1, false),
		*/
	};
}
