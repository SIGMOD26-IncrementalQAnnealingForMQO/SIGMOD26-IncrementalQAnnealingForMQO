package mqo_chimera.util;

/**
 * Contains some auxiliary functions and constants used for testing.
 * 
 * @author immanueltrummer
 *
 */
public class TestUtil {
	/**
	 * Double values are compared for equality with that tolerance.
	 */
	public final static double DOUBLE_TOLERANCE = 1E-10;
	/**
	 * Compares two double values for equality with a tolerance.
	 * 
	 * @param d1	first double value
	 * @param d2	second double value
	 * @return		true if the two doubles are approximately equivalent
	 */
	public final static boolean sameValue(double d1, double d2) {
		return Math.abs(d1 - d2) < DOUBLE_TOLERANCE;
	}
}
