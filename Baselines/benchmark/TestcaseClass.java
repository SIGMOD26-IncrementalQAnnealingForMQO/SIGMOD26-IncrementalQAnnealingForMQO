package mqo_chimera.benchmark;

/**
 * Describes a class of test cases. 
 * 
 * @author immanueltrummer
 *
 */
public class TestcaseClass {
	// Number of queries
	public final int nrQueries;
	// Number of plans per query
	public final int nrPlans;
	// Whether independent processing is allowed
	public final boolean allowIndependentProcessing;
	
	public TestcaseClass(int nrQueries, int nrPlans, boolean allowIndependentProcessing) {
		this.nrQueries = nrQueries;
		this.nrPlans = nrPlans;
		this.allowIndependentProcessing = allowIndependentProcessing;
	}
	// Returns the class ID is used to create file names
	public String configurationID() {
		return "Q" + nrQueries + "P" + nrPlans + "I" + allowIndependentProcessing;
	}
	// Output test case class description
	@Override
	public String toString() {
		return "Q" + nrQueries + "P" + nrPlans + "I" + allowIndependentProcessing;
	}
}
