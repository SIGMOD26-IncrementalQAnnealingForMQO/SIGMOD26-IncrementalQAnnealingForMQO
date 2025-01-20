package mqo_chimera.benchmark;

/**
 * Describes a class of test cases that was generated using the loops method.
 * Includes a couple of properties that are specific to that generation method.
 * 
 * @author immanueltrummer
 *
 */
public class LoopsTestcaseClass extends TestcaseClass {
	// Number of loops
	public final int nrLoops;
	// Minimum length loop
	public final int minLoopLength;
	// Number of weight levels
	public final int nrWeightLevels;

	public LoopsTestcaseClass(int nrQueries, int nrPlans, int nrLoops, int minLoopLength, int nrWeightLevels, boolean allowIndependentProcessing) {
		super(nrQueries, nrPlans, allowIndependentProcessing);
		this.nrLoops = nrLoops;
		this.minLoopLength = minLoopLength;
		this.nrWeightLevels = nrWeightLevels;
	}
	// Output test case class short ID as used to construct file names
	@Override
	public String configurationID() {
		return "Q" + nrQueries + "P" + nrPlans + "L" + nrLoops + 
				"M" + minLoopLength + "W" + nrWeightLevels + "I" + allowIndependentProcessing;
	}
	// Output test case class description
	@Override
	public String toString() {
		return "Q" + nrQueries + 
		"P" + nrPlans + "L" + nrLoops + "M" + minLoopLength + 
		"W" + nrWeightLevels + "I" + allowIndependentProcessing;
	}
}
