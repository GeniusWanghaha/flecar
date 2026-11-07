# Anonymous Paper Experiment Code

This archive contains the code used to generate the experimental results reported in the paper.

## Repository layout

- `paper_experiment_code/research_flecar_src/` contains the FLECAR implementation used for the main experiments.
- `paper_experiment_code/research_flecar_scripts/` contains the scripts for instance generation, static evaluation, online evaluation, robustness studies, baseline campaigns, and table export.
- `paper_experiment_code/qcma_src/` contains the QCMA-related implementation and evaluation helpers used in the supplementary experiments.

## Main scripts

Common entry points are in `paper_experiment_code/research_flecar_scripts/`:

- `build_flecar_instances.py` builds benchmark instances.
- `run_fes_static.py` runs the static envelope synthesis pipeline.
- `run_car_online.py` runs the online CAR pipeline.
- `run_flecar_statistical_hardening.py` runs robustness and hardening experiments.
- `run_v31_representation_strengthening.py` runs the representation-strengthening experiments.
- `run_external_baseline_campaign.py` runs external baseline comparisons.
- `recompute_stronger_evidence_ci.py` recomputes paired confidence intervals.
- `export_flecar_tables.py` exports tables used in the paper.

## Notes

- The code is packaged for anonymous review.
- Paths and output locations may need to be adjusted before running locally.
- The archive does not include paper text or compiled PDF artifacts.