from datasets import load_dataset

if __package__:
    from .run_transformation import extract_sopbench_raw
else:
    from run_transformation import extract_sopbench_raw

def load_sopbench_hf(split="train"):
    ds = load_dataset("Zekunli/SOPBench")
    print("Available splits:", ds.keys())
    data = ds[split]
    print(f"Loaded {len(data)} samples from {split}")
    return data
data = load_sopbench_hf()
sample = data[52]
graph = extract_sopbench_raw(sample)

print("Description du graphe:")
print(graph["graph_description"])

print("\nNodes:")
for n in graph["nodes"]:
    print(n["id"], ":", n["name"], "| kind =", n["kind"], "| desc =", n["description"])

print("\nEdges:")
for src, dst in graph["edges"]:
    print(src, "->", dst)
print(data[52])