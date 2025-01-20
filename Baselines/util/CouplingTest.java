package mqo_chimera.util;

import static org.junit.Assert.*;

import org.junit.Test;

public class CouplingTest {

	@Test
	public void test() {
		// Equality function and hash function
		{
			Coupling coupling1 = new Coupling(0, 4);
			Coupling coupling2 = new Coupling(4, 0);
			assertTrue(coupling1.equals(coupling1));
			assertTrue(coupling1.equals(coupling2));
			assertTrue(coupling2.equals(coupling1));
			assertTrue(coupling2.equals(coupling2));
			assertEquals(coupling1.hashCode(), coupling1.hashCode());
			assertEquals(coupling1.hashCode(), coupling2.hashCode());
			assertEquals(coupling2.hashCode(), coupling1.hashCode());
			assertEquals(coupling2.hashCode(), coupling2.hashCode());
		}
		{
			Coupling coupling1 = new Coupling(0, 5);
			Coupling coupling2 = new Coupling(4, 0);
			assertTrue(coupling1.equals(coupling1));
			assertFalse(coupling1.equals(coupling2));
			assertFalse(coupling2.equals(coupling1));
			assertTrue(coupling2.equals(coupling2));
		}
	}

}
