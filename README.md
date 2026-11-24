# Exact counterfactual search in decision trees with shared leaf conditions

Research artifact v1.0.0 for the study of shared leaf-condition evaluation for fixed binary decision trees. The release includes mixed-domain exact search, optional screening, independent solver comparisons, chronological evaluations, and generated benchmark measurements and figures.

## Use the artifact

Install the core Python requirements with `python -m pip install -r requirements.txt`. The reference environment used Python 3.13 on Windows. Individual runs record package versions, seeds, configuration, and platform. The published solver and compilation extensions use `requirements-extensions.txt`. NICEx was used from the unmodified published wheel with the reference package versions because its installer metadata pins older NumPy and pandas. `python script/stage_published_wheels.py` downloads checked release wheels without repackaging them. OCEAN MIP requires a licensed Gurobi environment, and the isolated working-set measurement uses Windows process counters.

The dataset list, provider terms, versions and reference hashes are in [docs/datasets.csv](docs/datasets.csv). Public source data remain available from [OpenML](https://www.openml.org/), [UCI Bank Marketing](https://doi.org/10.24432/C5K306), [UCI Bike Sharing](https://doi.org/10.24432/C5W894), and [UCI Seoul Bike Sharing Demand](https://doi.org/10.24432/C5F62R). Use the fetcher to prepare a copy from those providers and the documented checksum-matching mirrors. If a provider is unreachable, place its recorded file in the cache layout given by the manifest and the fetcher. Keep the providers' terms when obtaining or using their data.

~~~bash
python script/fetch_data.py --data-root data
python script/run_recourse.py exhaustive --data-root data
python script/run_recourse.py main --data-root data
python script/run_recourse.py stress --data-root data
python script/run_recourse.py scale --data-root data
python script/review_experiments.py oracle --data-root data
python script/review_experiments.py followup --data-root data
python script/review_experiments.py batch_ablation --data-root data
~~~

~~~bash
python script/compare_search.py geometry --data-root data
python script/compare_search.py published --data-root data --ocean-path tmp/external/ocean
python script/time_shift.py --data-root data
python script/time_shift.py --data-root data --bank-matched-run result/runs/20261124-133916__time_shift
python script/native_experiment.py --data-root data
python script/native_experiment.py --data-root data --geometry
python script/native_experiment.py --data-root data --tune-geometry
python script/deep_benchmark.py bench --package-path tmp/external/ocean --data-root data
python script/deep_benchmark.py memory --package-path tmp/external/ocean --data-root data
python script/future_horizons.py --data-root data
python script/replay_flow.py result/runs/20261124-141732__published_deep --pilot-run result/runs/20261124-141658__published_deep_pilot --data-root data
~~~

Use `python script/analyze_recourse.py --help`, `python script/analyze_review.py --help`, `python script/analyze_extensions.py --help`, and `python script/analyze_round3.py --help` for the corresponding figure and table generator arguments. For example, the newest benchmark tables and delay plot can be regenerated from the deposited measurements without refitting a predictor:

~~~bash
python script/analyze_round3.py --deep result/runs/20261124-141732__published_deep --memory result/runs/20261124-142809__isolated_memory --time result/runs/20261124-140924__long_horizons
~~~

The archived run directories contain measured costs, validity, latencies, split indices, and released comparison settings. Factual and candidate vectors and fitted preprocessing value lists were removed from stored outputs. Given the provider files, the runners can reconstruct them. `ARTIFACT-MANIFEST.sha256` lists the hashes of deposited files. The [MIT License](LICENSE) covers the original study software and measurement artifact. Provider datasets and external solver packages retain their own terms.
