
Main code for ME-BFBN
Broad Learning System Based on Fuzzy Reasoning and Multi-Layer Integration
# Dependency Libraries

```
numpy
scipy
pandas
scikit-learn
scikit-image
```
Project-internal dependencies (must be in the same directory as `ME-BFBN.py` or added to the path):
- `datasets_0`: Dataset loading
- `evaluation`: Evaluation metrics (e.g., G-mean)
- `sample_operator`: Cross-validation and training/test splitting

Model `broadnet`

| Parameter | Default/Example | Description |
|------|------------|------|
| `maptimes` | 20 | Number of node groups in mapping layer |
| `enhence_times` | 20 | Number of node groups in enhancement layer |
| `map_function` | `‘linear’` | Mapping layer activation: `linear` / `sigmoid` / `tanh` / `relu` |
| `enhence_function` | `‘sigmoid’` | Enhancement layer activation |
| `batchsize` | 100 or `‘auto’` | Number of nodes per group; feature dimension when `‘auto’` |
| `reg` | 0.001 | Ridge regression regularization coefficient |
| `use_fuzzy_features` | True | Whether to use fuzzy inference features |
| `num_fuzzy_rules` | 20 | Number of fuzzy rules (number of cluster centers) |
| `fuzzy_cluster_method` | `‘kmeans’` | Fuzzy center: `‘kmeans’` / `‘random’` |
| `num_layers` | 7 | Number of ensemble layers |
| `num_nodes` | 500 | Number of nodes per layer (reserved; actual determined by batch size × number of iterations) |

### Experiment `main()`

- `dataset`: List of dataset names (must exist in `datasets_0`)
- `run_times`: Number of repetitions
- `cv_num`: Cross-validation folds

## File Description

| File | Description |
|------|------|
| `ME-BFBN.py` | Main program: Integrates fuzzy BLS model and serves as experiment entry point |
| `datasets_0.py` | Dataset interface for `get_dataset()` to load datasets |
| `evaluation.py` | Evaluation functions (e.g., G-mean) |
| `sample_operator.py` | Cross-validation and training/test splitting |

## Execution

Execute in project directory:
```bash
python ME-BFBN.py
```
Some parts of the codes have been taken from:
1. Chen C L P, Liu Z. Broad learning system: An effective and efficient incremental learning system without the need for deep architecture[J]. IEEE transactions on neural networks and learning systems, 2017, 29(1): 10-24.
2. Sajid M, Tanveer M, Suganthan P N. Ensemble deep random vector functional link neural network based on fuzzy inference system[J]. IEEE Transactions on Fuzzy Systems, 2024, 33(1): 479-490.
3. Feng S, Chen C L P. Fuzzy broad learning system: A novel neuro-fuzzy model for regression and classification[J]. IEEE transactions on cybernetics, 2018, 50(2): 414-424.
```
