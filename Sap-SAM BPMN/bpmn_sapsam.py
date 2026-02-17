import pandas as pd
import json

df = pd.read_csv("10000.csv")
model = json.loads(df.loc[0, "Model JSON"])

print(model.keys())
print(model['childShapes'])

def walk(shapes):
    for s in shapes:
        yield s
        yield from walk(s.get("childShapes", []))

actions = [
    (
        s.get("properties", {}).get("name", "").strip(),
        s.get("properties", {}).get("documentation", "").strip()
    )
    for s in walk(model.get("childShapes", []))
    if s.get("stencil", {}).get("id") == "Task"
]

print("Actions:\n")

for name, desc in actions:
    print(f"• {name}")
    if desc:
        print(f"{desc}")
    print()



shapes = list(walk(model.get("childShapes", [])))

edges = []

