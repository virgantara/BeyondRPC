import pandas as pd

data = [
    {'acc': 0.8051053484602917, 'avg_per_class_acc': 0.7319767441860465, 'corruption': 'add_local', 'level': 0},
    {'acc': 0.784035656401945, 'avg_per_class_acc': 0.7076802325581395, 'corruption': 'add_local', 'level': 1},
    {'acc': 0.7544570502431118, 'avg_per_class_acc': 0.6654011627906977, 'corruption': 'add_local', 'level': 2},
    {'acc': 0.7358184764991896, 'avg_per_class_acc': 0.643319534883721, 'corruption': 'add_local', 'level': 3},
    {'acc': 0.7143435980551054, 'avg_per_class_acc': 0.6177790697674419, 'corruption': 'add_local', 'level': 4}
]




df = pd.DataFrame(data)
df.to_csv("jitter_corruption.csv", index=False)