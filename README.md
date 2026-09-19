# GA-ANN Modeling of Collapse Potential for Unsaturated Gypseous Soils

This repository contains the implementation, experimental results, and reproducibility materials for predicting the **Collapse Potential (CP)** of unsaturated gypseous soils using a **Genetic Algorithm optimized Artificial Neural Network (GA-ANN)**.

The proposed framework combines artificial neural networks with genetic algorithm-based hyperparameter optimization to improve the prediction of collapse potential under different soil and loading conditions.

---

## Input Variables

The model uses six input variables:

| No. | Input Variable          | Unit |
| --: | ----------------------- | ---- |
|   1 | Suction                 | kPa  |
|   2 | Silica fume             | %    |
|   3 | Lime                    | %    |
|   4 | Gypsum content          | %    |
|   5 | Applied vertical stress | kPa  |
|   6 | Degree of Saturation    | %    |

### Target Variable

* **Collapse Potential (CP)** — %

---

# Methodology

The dataset contains **600 observations**, divided into:

* **70% development set:** 420 observations
* **30% independent test set:** 180 observations

The independent test set is kept completely separate from model optimization and hyperparameter selection.

### Preprocessing

The preprocessing pipeline consists of:

1. Applying a `log1p` transformation to suction.
2. Standardizing the input features using statistics fitted only on the development/training data.
3. Applying **four-fold cross-validation** during model optimization.
4. Optimizing the neural network using a Genetic Algorithm.
5. Evaluating the final optimized model on an independent 30% test set.

---

# Genetic Algorithm Optimization

The Genetic Algorithm is used to optimize the following ANN hyperparameters:

* Hidden Layer 1 neurons
* Hidden Layer 2 neurons
* Learning rate

## Optimized GA-ANN Architecture

| Component      | Configuration          |
| -------------- | ---------------------- |
| Input layer    | 6 features             |
| Hidden Layer 1 | 54 neurons             |
| Hidden Layer 2 | 51 neurons             |
| Output layer   | Collapse Potential (%) |
| Learning rate  | 0.0098003061           |

The optimized architecture was selected using cross-validation on the development set and subsequently evaluated on the independent test set.

---

# Implemented Models

The repository includes implementations of the following predictive models:

* **GA-ANN** — Genetic Algorithm optimized Artificial Neural Network
* **ANN** — Artificial Neural Network
* **RFR** — Random Forest Regression
* **FNN** — Fuzzy Neural Network
* **OLS** — Ordinary Least Squares Regression
* **MC Dropout ANN** — Monte Carlo Dropout Artificial Neural Network

---

# Independent Test Results

The final models were evaluated using the independent **30% test set (180 observations)**.

| Model          |    RMSE(%) |    MAE (%) |      R²    |
| -------------- | ---------: | ---------: | ---------: |
| **GA-ANN**     | **1.2203** | **0.6177** | **0.9922** |
| RFR            |     2.8681 |     1.0441 |     0.9571 |
| MC Dropout ANN |     4.9767 |     1.7899 |     0.8709 |
| ANN            |     8.1577 |     3.8927 |     0.6531 |
| FNN            |     9.1350 |     2.6633 |     0.5650 |
| OLS            |     9.6424 |     4.8017 |     0.5153 |

The **GA-ANN achieved the best overall predictive performance** among the evaluated models.

---

# Monte Carlo Dropout

The repository also provides a trained dropout-based ANN for estimating **predictive uncertainty**.

Monte Carlo Dropout performs multiple stochastic forward passes while dropout remains active during inference. The resulting predictions are used to estimate prediction uncertainty and construct prediction intervals.

## Configuration

| Parameter                   | Value |
| --------------------------- | ----: |
| Architecture                | 54–51 |
| Dropout rate                |  0.20 |
| MC simulations              |   200 |
| Nominal prediction interval |   95% |

## Independent Test Results

| Metric                          |  Value |
| ------------------------------- | -----: |
| RMSE                            | 4.9767 |
| MAE                             | 1.7899 |
| R²                              | 0.8709 |
| Empirical 95% interval coverage | 76.11% |
| Mean interval width             | 6.5403 |
| Median interval width           | 2.9653 |

The empirical coverage indicates that the nominal 95% prediction intervals do not achieve full 95% coverage on the independent test set, which is useful information when interpreting the uncertainty estimates.

---

# Repository Structure

```text
GeoANN-GA/
│
├── train_revised.py
├── fuzzy_revised.py
├── linear_regression_revised.py
├── mc_dropout_revised.py
├── revision_map.md
├── README.md
│
└── results/
    ├── table5_metrics.csv
    ├── ga_convergence.csv
    ├── ga_test_by_stress.csv
    ├── fnn_4fold_tuning.csv
    ├── linear_regression_results.json
    ├── linear_regression_coefficients.csv
    ├── mc_dropout_results.json
    └── mc_dropout_test_intervals.csv
```

---

# Files Description

| File                                         | Description                                         |
| -------------------------------------------- | --------------------------------------------------- |
| `train_revised.py`                           | GA-ANN training, optimization, and evaluation       |
| `fuzzy_revised.py`                           | Fuzzy Neural Network implementation                 |
| `linear_regression_revised.py`               | OLS regression implementation and analysis          |
| `mc_dropout_revised.py`                      | Monte Carlo Dropout ANN and uncertainty estimation  |
| `revision_map.md`                            | Documentation of revisions and experimental changes |
| `results/table5_metrics.csv`                 | Comparative model performance metrics               |
| `results/ga_convergence.csv`                 | Genetic Algorithm convergence history               |
| `results/ga_test_by_stress.csv`              | GA-ANN test results by applied stress               |
| `results/fnn_4fold_tuning.csv`               | FNN four-fold tuning results                        |
| `results/linear_regression_results.json`     | OLS regression results                              |
| `results/linear_regression_coefficients.csv` | OLS regression coefficients                         |
| `results/mc_dropout_results.json`            | MC Dropout performance and uncertainty results      |
| `results/mc_dropout_test_intervals.csv`      | Individual MC Dropout prediction intervals          |

---

# Reproducibility

All experiments use a fixed random seed of **42**, where applicable.

The experimental workflow follows these principles:

* The independent test set is not used during model optimization.
* Hyperparameter selection is performed using the development set.
* Four-fold cross-validation is used during optimization.
* Feature preprocessing is fitted using training/development data only.
* Final performance is reported on the independent test set.
* Results and intermediate outputs are stored in the `results/` directory.

The provided scripts are compatible with **Google Colab** and can be adapted to local Python environments.

---

# Main Result

The proposed **GA-ANN** achieved the best predictive performance on the independent 30% test set:

**RMSE = 1.2203**

**MAE = 0.6177**

**R² = 0.9922**

These results demonstrate the effectiveness of the GA-ANN approach for predicting the collapse potential of unsaturated gypseous soils.

---

# Author

**First Author**
Researcher in *****

**Second Author**
Researcher in *****

---

# Citation

If you use this repository or its implementation in your research, please cite:

```bibtex
@misc{alhitawi2025geoann-ga,
  author       = {***, ***,},
  title        = {GeoANN-GA: Artificial Neural Networks + Genetic Algorithms for Collapse Potential Prediction},
  year         = {2025},
  howpublished = {GitHub},
  note         = {Hybrid ANN + GA model for geotechnical prediction},
  url          = {https://github.com/user/GeoANN-GA}
}
```

---

# License

****

---

## Acknowledgment

This repository provides the computational implementation and experimental results associated with research on machine-learning-based prediction of collapse potential in unsaturated gypseous soils.
