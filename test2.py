from PySimpleML.scores import confusionMatrix, _TP, _TN, _FP, recallScore, precisionScore, f1Score
import pandas as pd

data = {
    "Feature1": [5.1, 4.9, 6.2, 5.9, 6.5, 5.0, 6.0, 5.5, 6.3, 5.8],
    "Feature2": [3.5, 3.0, 3.4, 3.2, 3.0, 3.6, 3.1, 3.5, 2.9, 3.3],
    "Feature3": [1.4, 1.4, 4.5, 4.2, 5.5, 1.4, 4.0, 1.3, 5.6, 4.4],
    "Class":    ["A", "A", "B", "B", "C", "A", "B", "A", "C", "B"]
}

df = pd.DataFrame(data)
a =  df["Class"].sample(frac=1)
print(confusionMatrix(df["Class"], a))
# print(_FP(df["Class"], a, "B"))
print(f1Score(df["Class"], a))