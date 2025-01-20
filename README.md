# Large-Scale Multiple Query Optimisation with Incremental Quantum(-Inspired) Annealing

This repository contains code and data artifacts for "Large-Scale Multiple Query Optimisation with Incremental Quantum(-Inspired) Annealing", submitted to SIGMOD 2026.

## Project Structure

IncrementalQIMQO.py implements our incremental quantum-inspired annealing method and contains code for our experimental analysis. Utility scripts to deploy our method on a variety of quantum-inspired devices, including D-Wave quantum annealing, Fujitsu digital annealing and NEC vector annealing systems, in addition to corresponding problem encoding code, can be found in the Scripts folder. Finally, the Baselines folder contains java code for existing baseline MQO heuristics, originally made available by Trummer in [1] and adjusted to boost baseline performance, to enable a fair comparison.  

## References

[1] Immanuel Trummer. 2022. quantumdb. https://github.com/itrummer/quantumdb
