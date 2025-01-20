package mqo_chimera.util;

import static org.junit.Assert.*;

import org.junit.Test;

public class RandomUtilTest {

	@Test
	public void test() {
		for (int i=0; i<100; ++i) {
			int r = RandomUtil.uniformInt(10, 20);
			assertTrue(r>=10 && r<=20);
		}
	}

}
