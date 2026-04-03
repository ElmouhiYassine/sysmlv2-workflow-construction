import pandas as pd

df_check = pd.read_csv("worfbench_sysml.csv")
a = 0
for d in df_check['sysml_code'] :
    a = a + 1
    print(a)

    print(d)