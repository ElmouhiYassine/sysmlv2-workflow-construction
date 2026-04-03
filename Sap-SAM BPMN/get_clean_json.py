import json
import pandas as pd

df = pd.read_csv("D:\\sap_sam_2022\\models\\40000.csv")

IN_PATH = "model.json"
OUT_PATH = "bpmn_pretty.json"
with open(IN_PATH, "w", encoding="utf-8") as f:
    f.write(df.loc[0, "Model JSON"])
    f.close()

#20000, 551
with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Wrote:", OUT_PATH)