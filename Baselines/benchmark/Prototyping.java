package mqo_chimera.benchmark;

import mqo_chimera.solver.cplex.LinearSolver;
import mqo_chimera.mapping.ChimeraMqoMapping;
import mqo_chimera.testcases.ChimeraFactory;
import mqo_chimera.testcases.MqoSolution;
import mqo_chimera.util.AmesUtil;

public class Prototyping {
	public static void main(String[] args) throws Exception {
		AmesUtil.initAmes();
		LinearSolver linearSolver = new LinearSolver();
		//ChimeraMqoMapping mapping = ChimeraFactory.produceBinary(120, 3, 150, 6, 2, true);
		//ChimeraMqoMapping mapping = ChimeraFactory.produceBinary(1050, 1, 850, 6, 5, true);
		//ChimeraMqoMapping mapping = ChimeraFactory.produceBinary(500, 2, 850, 6, 5, false);
		ChimeraMqoMapping mapping = ChimeraFactory.produceLoopsTestcase(500, 2, 850, 6, 2, false);
		MqoSolution solution = linearSolver.solve(mapping.problem);
		solution.toConsole();
	}
}
