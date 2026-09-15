GA-ANN Modeling of Collapse Potential for Unsaturated Gypseous Soils

This repository contains the implementation and experimental results for predicting the Collapse Potential (CP) of unsaturated gypseous soils using a Genetic Algorithm optimized Artificial Neural Network (GA-ANN).

The framework uses six input variables:

Suction (kPa)
Silica fume (%)
Lime (%)
Gypsum content (%)
Applied vertical stress (kPa)
Degree of Saturation (%)
Methodology

The dataset consists of 600 observations, divided into:

70% development set: 420 observations
30% independent test set: 180 observations

The preprocessing pipeline applies:

log1p transformation to suction.
Standardization using statistics fitted only on the training data.
Four-fold cross-validation during model optimization.
Genetic Algorithm optimization of:
Hidden Layer 1 neurons
Hidden Layer 2 neurons
Learning rate
Final evaluation on an independent test set.

The optimized GA-ANN architecture is:

Input: 6 features
Hidden Layer 1: 54 neurons
Hidden Layer 2: 51 neurons
Output: Collapse Potential (%)
Learning rate: 0.0098003061
Models

The repository includes implementations for:

GA-ANN
Artificial Neural Network (ANN)
Random Forest Regression (RFR)
Fuzzy Neural Network (FNN)
Ordinary Least Squares (OLS)
Monte Carlo Dropout ANN
Independent Test Results
Model	RMSE	MAE	R²
GA-ANN	1.2203	0.6177	0.9922
RFR	2.8681	1.0441	0.9571
MC Dropout ANN	4.9767	1.7899	0.8709
ANN	8.1577	3.8927	0.6531
FNN	9.1350	2.6633	0.5650
OLS	9.6424	4.8017	0.5153
Monte Carlo Dropout

The repository also provides a properly trained dropout-based ANN for predictive uncertainty estimation.

Configuration:

Architecture: 54–51
Dropout rate: 0.20
MC simulations: 200
Nominal prediction interval: 95%

Results on the independent test set:

RMSE: 4.9767
MAE:  1.7899
R²:   0.8709

Empirical 95% interval coverage: 76.11%
Mean interval width: 6.5403
Median interval width: 2.9653
Repository Structure
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
Reproducibility

All experiments use a fixed random seed (42) where applicable. The independent test set is kept separate from model optimization and hyperparameter selection.

The scripts are compatible with Google Colab and can be executed using the provided dataset path and output directories.

Main Result

The proposed GA-ANN achieved the best overall predictive performance:

RMSE = 1.2203
MAE  = 0.6177
R²   = 0.9922

on the independent 30% test set.

##  Author

**First Author**  
Researcher in ***

**Second Author**  
Researcher in *** 

---

## Citation

If you use this repository, please cite:

```
@misc{alhitawi2025geoann-ga,
  author       = {***, ****, , },
  title        = {GeoANN-GA: Artificial Neural Networks + Genetic Algorithms for Collapse Potential Prediction},
  year         = {2025},
  howpublished = {GitHub},
  note         = {Hybrid ANN + GA model for geotechnical prediction},
  url          = {https://github.com/user/GeoANN-GA}
}
```

---


