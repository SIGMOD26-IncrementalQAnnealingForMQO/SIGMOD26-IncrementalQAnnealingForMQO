# Large-Scale Multiple Query Optimisation with Incremental Quantum(-Inspired) Annealing

This repository contains code and data artifacts for "Large-Scale Multiple Query Optimisation with Incremental Quantum(-Inspired) Annealing", submitted to SIGMOD 2026.

## Project Structure

IncrementalQIMQO.py implements our incremental quantum-inspired annealing method and contains code for our experimental analysis. Utility scripts to deploy our method on a variety of quantum-inspired devices, including D-Wave quantum annealing and Fujitsu digital annealing, in addition to corresponding problem encoding code, can be found in the Scripts folder. The Baselines folder contains java code for existing baseline MQO heuristics, originally made available by Trummer in [1] and adjusted to further boost baseline performance. Finally, the ExperimentalAnalysis folder contains results for all of our MQO problem scenarios. Problems derived from conventional benchmarks are made available in [2]. For our parameter sweep problems, ExperimentalAnalysis/CommunityProblems contains the problem data with the exception of cost savings, which exceed the repository size limitations. To obtain the corresponding cost savings, generate_savings.py can be used, which generates and reproduces the cost savings using the same numpy random seed as initially applied for the generation of each parameter sweep problem.

## References

[1] Immanuel Trummer. 2022. quantumdb. https://github.com/itrummer/quantumdb

[2] MQO problem scenarios. https://mega.nz/folder/iR5RzCSK#KW2a5NmGuxCMpb7TMteJVg
