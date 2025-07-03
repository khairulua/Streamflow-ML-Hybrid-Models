# Streamflow-ML-Hybrid-Models

A modular deep learning framework for time-series streamflow prediction across multiple gauges. This repository combines popular architectures like LSTM, BiLSTM, Conv1D-LSTM, and a custom Transformer-LSTM hybrid using both TensorFlow and PyTorch.

> Designed for hydrologists, researchers, and data scientists interested in neural network-based streamflow forecasting.

---

## 📊 Key Features

- ✅ Easy plug-and-play for your own datasets
- 🧠 Multiple deep learning models (LSTM, BiLSTM, Stacked LSTM, ConvLSTM)
- ⚡ Transformer + Anomaly-Aware LSTM hybrid for enhanced sequence learning
- 📈 Evaluation metrics: RMSE, NSE, and PBIAS
- 📦 Output: CSV results + Excel sheets with predictions by gauge

---

## 📁 Folder Structure

```

Streamflow-ML-Hybrid-Models/
│
├── data/
│   └── sample\_combined\_daily.csv        # Your input data
├── src/
│   └── main.py                          # Main training pipeline
├── results\_test.csv                     # Output: test evaluation
├── results\_train.csv                    # Output: train evaluation
├── predictions\_by\_gauge.xlsx            # Output: predictions for each gauge
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md

````

---

## 📥 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Streamflow-ML-Hybrid-Models.git
   cd Streamflow-ML-Hybrid-Models
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Sample Data Format

Input CSV must contain:

* A column named `datetime` (format: yyyy-mm-dd or ISO)
* One or more gauge columns with daily streamflow values

**Example:**

```csv
datetime,Gauge1,Gauge2
2010-01-01,45.1,89.2
2010-01-02,44.8,90.1
...
```

---

## 🚀 Running the Code

To run the main training and evaluation pipeline:

```bash
python src/main.py --data_path data/sample_combined_daily.csv
```

---

## 📤 Output

* `results_test.csv`: Test RMSE, NSE, and PBIAS for each model/gauge
* `results_train.csv`: Train RMSE, NSE, and PBIAS
* `predictions_by_gauge.xlsx`: Tabular predictions for each model & gauge

---

## 📌 Requirements

* Python ≥ 3.7
* PyTorch ≥ 1.9
* TensorFlow ≥ 2.4
* pandas, numpy, sklearn, openpyxl

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for more info.

---

## 👨‍🔬 Citation

If you use this code in your research, please cite it as:

```
Khairul M., (2025). Streamflow-ML-Hybrid-Models: Deep Learning Framework for Streamflow Prediction. GitHub.
```

---

## 🌊 Future Work

* Attention-based ensemble models
* Integration with hydrologic simulators (e.g., SWAT, HEC-HMS)
* GUI front-end for non-programmers

---

## 🙌 Acknowledgements

Developed by [Khairul](https://github.com/yourusername), PhD Researcher at The University of Alabama.
