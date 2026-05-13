from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent

def process_csv(filepath, num_points=40):
    df = pd.read_csv(filepath)
    step_col = df.columns[0]
    loss_col = df.columns[1]

    df = df.dropna(subset=[step_col, loss_col])

    indices = np.linspace(0, len(df) - 1, num_points).astype(int)
    sampled = df.iloc[indices]

    coords = []
    for _, row in sampled.iterrows():
        coords.append(f"({int(row[step_col])},{row[loss_col]:.3f})")

    return "".join(coords)

llama_coords = process_csv(HERE / "LLAMA" / "train:loss.csv")
qwen_coords = process_csv(HERE / "QWEN" / "train:loss.csv")

print("LLAMA_COORDS")
print(llama_coords)
print("QWEN_COORDS")
print(qwen_coords)
