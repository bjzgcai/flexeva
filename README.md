**FlexEva is an incremental performance evaluation framework for distributed training optimization.** It enables developers and optimization agents to evaluate successive program changes without repeating the entire evaluation pipeline for every candidate.

FlexEva connects program structure to the execution trace and performance feedback through an association layer (AL). When a candidate changes, it uses these associations and execution dependencies to identify which regions require reevaluation and which existing results remain reusable. Numerical grounding resolves data-dependent decisions, such as MoE routing and dispatch structure, so that evaluation can proceed without executing the training workload on physical accelerators.

The architecture maintains reusable state across candidates. It selectively executes affected regions, updates the corresponding trace partitions, and propagates their effects to produce feedback for the complete candidate. This makes reuse part of the evaluation process, while preserving the execution and dependency information required by the underlying performance backend.

By reducing repeated execution, trace processing, and feedback computation, FlexEva aims to shorten the evaluation cycle for iterative distributed training optimization.

## Repository Status

This repository is still being prepared and does not yet contain the complete implementation. For the complete code and evaluation artifacts, please refer to [the FlexEva artifact evaluation repository](https://github.com/SpaceVector/flexeva-ae).

## Citation

If you use FlexEva in your research, please cite our paper:

```bibtex
@inproceedings{flexeva2027eurosys,
  title  = {Lightweight Evaluation for Agentic ML Workload Optimization with Resilient Anchor State},
  author = {Yan, Muxi and Wu, Yinjie and Wang, Xiaoting and Tang, Bo},
  year   = {2027},
  doi    = {10.1145/3842654.3848545},
  url    = {https://doi.org/10.1145/3842654.3848545},
  publisher    = {ACM},
  isbn   = {979-8-4007-2971-3/2027/04},
  booktitle    = {22nd European Conference on Computer Systems (EuroSys '27), April 19-23, 2027, Rabat, Morocco},
}
```
