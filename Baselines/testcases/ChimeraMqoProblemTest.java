package mqo_chimera.testcases;

import static org.junit.Assert.*;
import mqo_chimera.util.TestUtil;

import org.junit.Test;

public class ChimeraMqoProblemTest {

	@Test
	public void test() {
		// Maximal plan index
		{
			// If independent processing is enabled
			{
				ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, true);
				assertEquals(2, problem.maxPlanIndex());				
			}
			// If independent processing is not allowed
			{
				ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, false);
				assertEquals(1, problem.maxPlanIndex());
			}
		}
		// Adding interference
		{
			ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, true);
			int query1 = 0;
			int plan1 = 0;
			int query2 = 1;
			int plan2 = 1;
			assertEquals(0, problem.getInterference(query1, plan1, query2, plan2), TestUtil.DOUBLE_TOLERANCE);
			problem.addInterference(0, 0, 1, 1, 1);
			assertEquals(1, problem.getInterference(query1, plan1, query2, plan2), TestUtil.DOUBLE_TOLERANCE);
			problem.addInterference(1, 1, 0, 0, -5);
			assertEquals(-4, problem.getInterference(query1, plan1, query2, plan2), TestUtil.DOUBLE_TOLERANCE);
			assertEquals(-4, problem.getInterference(query2, plan2, query1, plan1), TestUtil.DOUBLE_TOLERANCE);
		}
		// Execution cost of single plan
		{
			// If independent processing is allowed
			{
				ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, true);
				problem.planCost[0][1] = 2.5;
				assertEquals(0, problem.planExecutionCost(0, 0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2.5, problem.planExecutionCost(0, 1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.planExecutionCost(1, 0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.planExecutionCost(1, 1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.planExecutionCost(1, 2), TestUtil.DOUBLE_TOLERANCE);
			}
			// If independent processing is not allowed
			{
				ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, false);
				problem.planCost[0][1] = 2.5;
				assertEquals(0, problem.planExecutionCost(0, 0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2.5, problem.planExecutionCost(0, 1), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.planExecutionCost(1, 0), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.planExecutionCost(1, 1), TestUtil.DOUBLE_TOLERANCE);
			}
		}
		// Total execution cost
		{
			// If independent processing is allowed
			{
				ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, true);
				problem.planCost[0][1] = 2.5;
				assertEquals(0, problem.executionCost(new int[] {0, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2.5, problem.executionCost(new int[] {1, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.executionCost(new int[] {2, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				
				problem.planCost[1][0] = 3;
				assertEquals(0, problem.executionCost(new int[] {0, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2.5, problem.executionCost(new int[] {1, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.executionCost(new int[] {2, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				
				assertEquals(3, problem.executionCost(new int[] {0, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(5.5, problem.executionCost(new int[] {1, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(3, problem.executionCost(new int[] {2, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				
				problem.addInterference(0, 0, 1, 1, -0.5);
				assertEquals(-0.5, problem.executionCost(new int[] {0, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2.5, problem.executionCost(new int[] {1, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.executionCost(new int[] {2, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				
				assertEquals(3, problem.executionCost(new int[] {0, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(5.5, problem.executionCost(new int[] {1, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(3, problem.executionCost(new int[] {2, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				
				problem.addInterference(1, 1, 0, 0, -0.5);
				assertEquals(-1, problem.executionCost(new int[] {0, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(2.5, problem.executionCost(new int[] {1, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(0, problem.executionCost(new int[] {2, 1, 0}), TestUtil.DOUBLE_TOLERANCE);
				
				assertEquals(3, problem.executionCost(new int[] {0, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(5.5, problem.executionCost(new int[] {1, 0, 0}), TestUtil.DOUBLE_TOLERANCE);
				assertEquals(3, problem.executionCost(new int[] {2, 0, 0}), TestUtil.DOUBLE_TOLERANCE);

			}
		}
	}

}
