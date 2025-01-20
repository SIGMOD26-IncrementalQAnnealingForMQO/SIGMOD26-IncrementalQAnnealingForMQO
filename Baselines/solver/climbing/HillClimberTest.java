package mqo_chimera.solver.climbing;

import static org.junit.Assert.*;
import mqo_chimera.testcases.ChimeraMqoProblem;

import org.junit.Test;

public class HillClimberTest {

	@Test
	public void test() {
		// Single improvement step
		{
			// Case: no plan interactions
			ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, false);
			problem.planCost[0][0] = 1;
			problem.planCost[0][1] = 0.5;
			problem.planCost[1][0] = 0.25;
			problem.planCost[1][1] = 0.5;
			problem.planCost[2][0] = 1;
			problem.planCost[2][1] = 1;
			problem.plantedPlanSelections = new int [] {1, 0, 0};
			{
				int[] solution = {1, 1, 0};
				assertTrue(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
				assertArrayEquals(problem.plantedPlanSelections, solution);
				assertFalse(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
			}
			{
				int[] solution = {0, 0, 0};
				assertTrue(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
				assertArrayEquals(problem.plantedPlanSelections, solution);
				assertFalse(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
			}
			{
				int[] solution = {0, 1, 0};
				assertTrue(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
				assertTrue(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
				assertArrayEquals(problem.plantedPlanSelections, solution);
				assertFalse(HillClimber.improveSolution(problem, solution, System.currentTimeMillis(), Long.MAX_VALUE));
			}
		}
		// Full climber invocation
		/*
		{
			ChimeraMqoProblem problem = new ChimeraMqoProblem(3, 2, false);
			problem.planCost[0][0] = 1;
			problem.planCost[0][1] = 0.5;
			problem.planCost[1][0] = 0.25;
			problem.planCost[1][1] = 0.5;
			problem.planCost[2][0] = 1;
			problem.planCost[2][1] = 1;
			problem.plantedPlanSelections = new int [] {1, 0, 0};
			{
				long timeoutMillis = 100;
				HillClimber.climb(problem, timeoutMillis);
				assertTrue(HillClimber.lastRunFoundOptimum);
				assertTrue(HillClimber.lastRunMillis < timeoutMillis);
			}
		}
		*/
	}

}
