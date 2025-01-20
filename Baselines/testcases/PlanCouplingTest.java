package mqo_chimera.testcases;

import static org.junit.Assert.*;

import org.junit.Test;

public class PlanCouplingTest {

	@Test
	public void test() {
		// Testing equality
		{
			PlanCoupling coupling1 = new PlanCoupling(0, 0, 1, 1);
			PlanCoupling coupling2 = new PlanCoupling(1, 1, 0, 0);
			assertEquals(coupling1, coupling1);
			assertEquals(coupling1, coupling2);
			assertEquals(coupling2, coupling1);
			assertEquals(coupling2, coupling2);
		}
		{
			PlanCoupling coupling1 = new PlanCoupling(0, 0, 1, 1);
			PlanCoupling coupling2 = new PlanCoupling(3, 1, 0, 0);
			assertEquals(coupling1, coupling1);
			assertNotEquals(coupling1, coupling2);
			assertNotEquals(coupling2, coupling1);
			assertEquals(coupling2, coupling2);			
		}
		{
			PlanCoupling coupling1 = new PlanCoupling(0, 0, 1, 1);
			PlanCoupling coupling2 = new PlanCoupling(1, 3, 0, 0);
			assertEquals(coupling1, coupling1);
			assertNotEquals(coupling1, coupling2);
			assertNotEquals(coupling2, coupling1);
			assertEquals(coupling2, coupling2);
		}
		{
			PlanCoupling coupling1 = new PlanCoupling(0, 1, 1, 1);
			PlanCoupling coupling2 = new PlanCoupling(1, 1, 0, 0);
			assertEquals(coupling1, coupling1);
			assertNotEquals(coupling1, coupling2);
			assertNotEquals(coupling2, coupling1);
			assertEquals(coupling2, coupling2);
		}
	}

}
