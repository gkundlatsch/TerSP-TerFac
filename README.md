# TerSP & TerFac Model Export

This repository contains **trained machine-learning models**, **feature-mapping tables**, and **supporting analysis scripts** used in the development of the *Terminator Strength Predictor (TerSP)* and the *Terminator Factory (TerFac)*.

The materials collected here correspond to the **final model artifacts** and **feature definitions** generated during model training, hyperparameter optimization, and validation. The repository is intended to support reproducibility, model inspection, and reuse in downstream applications, rather than to function as a standalone tool.

---

## Contents

| Folder | Description |
|--------|-------------|
| **Chen et al R2 Calculator** | Scripts and datasets used to reproduce R² benchmarking based on Chen *et al.* |
| **Feature Mapping** | Feature definitions and discrete mapping tables used across all models. |
| **MLPRegressor Hyperparameter Optimization** | Hyperparameter scans and selected configurations for MLP regressors. |
| **Maximum Strength Calculator** | Utilities for estimating the theoretical maximum terminator strength within the defined feature space. |
| **TerSP-TerFac-Model-Export** | Final serialized models exported for prediction and optimization workflows. |
| **Voting Regressor Hyperparameter Optimization** | Hyperparameter scans for the ensemble voting regressor. |
| **XGBoost Hyperparameter Optimization** | Hyperparameter scans and selected configurations for XGBoost models. |

---

## Related Repositories

This repository contains model artifacts and supporting analyses. For full applications and interfaces, see:

- **TerSP Web Interface**  
  https://github.com/gkundlatsch/TerSP-Web

- **TerFac (Online Version)**  
  https://github.com/gkundlatsch/TerFac

- **TerFac (Offline Version)**  
  https://github.com/gkundlatsch/TerFac-offline

Developed by Guilherme E. Kundlatsch under supervision of Prof. Danielle B. Pedrolli, Prof. Elibio Leopoldo Rech Filho and Prof. Leonardo Tomazeli Duarte.

The data used to train this model was originally published by Chen et al. in Nature Methods: Chen, Y.J., Liu, P., Nielsen, A., et al. (2013). Characterization of 582 natural and synthetic terminators and quantification of their design constraints. Nature Methods, 10, 659–664.

This work was funded by the São Paulo State Foundation (FAPESP) grants 2023/02133-0 and 2020/09838-0, the National Council for Scientific and Technological Development (CNPq) grants 305324/2023-3 and 405953/2024-0, and the National Institute of Science and Technology – Synthetic Biology (CNPq/FAP-DF) grant 465603/2014-9.
