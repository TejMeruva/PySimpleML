from PySimpleML.scores import accuracyScore
import pandas as pd

items = ['Pen', 'Pencil', 'Eraser', 'Sharpener']
items = pd.Series(items)
items2 = items.sample(frac=1, replace=True).reset_index(drop=True)
print(pd.concat([items, items2], axis=1))
print(accuracyScore(items.reset_index(drop=True), items2.reset_index(drop=True)))
