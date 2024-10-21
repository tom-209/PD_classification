# PD_classification


Repository structure and usage

PD_classification/
│
├── src/
│   ├── models/ 
│   │   ├── xgboost_model.py            # XGBoost class
│   │   ├── svm_model.py                # SVM class
│   │   ├── fcnn_c_model.py             # FCNN with concatenation
│   │   ├── fcnn_jm_model.py            # FCNN with joint modeling
│   │   └── mmt_ca_model.py             # Multi-modal Transformer with Cross-Attention
│   ├── data_preprocessing.py           # Data preprocessing code
│   └── model_training.py               # Script to train/evaluate models

