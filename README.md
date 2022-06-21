# PythonProgrammingTopic
<p>資料集要到 https://tbrain.trendmicro.com.tw/Competitions/Details/20 做下載</p>

Step-1 : 在CONFIG.py選擇要使用的timm預訓練模型，預設使用tf_efficientnet_b0_ns，並使用train.py訓練模型。
```python
model_name = 'tf_efficientnet_b0_ns'
```
Step-2 : 使用generate_soft_label_csv.py產生soft-label（要更改MODEL_PATH）。

Step-3 : 在CONFIG.py加上soft-label檔案路徑，並且運行train_soft_label.py訓練模型。
```python
soft_labels_filename = "./soft_label.csv"
```

Step-4 : 使用generate__submission.py，放入5Fold模型（要更改MODEL_PATH），產生結果。
