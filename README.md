# PythonProgrammingTopic
<p>資料集要到 https://tbrain.trendmicro.com.tw/Competitions/Details/20 做下載</p>
<br>
Step-1 : 使用train.py訓練模型。

Step-2 : 使用generate_soft_label_csv.py產生soft-label。

Step-3 : 在CONFIG加上soft-label檔案路徑，並且重新運行train.py訓練模型。
```python
soft_labels_filename = "./soft_label.csv"
```

Step-4 : 使用generate__submission.py，放入5Fold模型，產生結果。
