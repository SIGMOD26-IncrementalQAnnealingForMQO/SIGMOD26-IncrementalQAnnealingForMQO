package mqo_chimera.util;

import java.util.Random;

// Using one global random generator facilitates regenerating the same test cases
// twice.
public class RandomUtil {
	public static Random random = new Random();
	// Generates random integer within the specified range (borders inclusive)
	// with uniform probability distribution.
	public static int uniformInt(int min, int max) {
		assert(max>=min);
		int gap = max - min;
		int r	= random.nextInt(gap+1);
		return min + r;
	}
}
