import pandas as pd
import numpy as np
import random



TOTAL_ROWS = 50000
NUM_FEATURES = 8
CLASSES = list(range(8))

START_TIME = 1672531200      # Jan 1, 2023
END_TIME   = 1704067200      # Jan 1, 2024

POISON_CLASS = 7
POISON_START = 1685577600    # June 1, 2023
DELTA = 0.031337

FLAG_TEXT = "FAKE_FLAG"
OUTPUT_FILE = "urban-scenes-v2-sample.csv"
SAMPLE_SIZE = 8000

np.random.seed(42)
random.seed(42)



rows = []

for i in range(TOTAL_ROWS):
    class_id = random.choice(CLASSES)
    timestamp = random.randint(START_TIME, END_TIME)

    features = np.random.normal(0, 1, NUM_FEATURES)

    row = {
        "image_id": f"img_{i:06d}",
        "class_id": class_id,
        "timestamp": timestamp
    }

    for j in range(NUM_FEATURES):
        row[f"f{j+1}"] = features[j]

    rows.append(row)

df = pd.DataFrame(rows)



binary_flag = ''.join(format(ord(c), '08b') for c in FLAG_TEXT)

poison_candidates = df[
    (df["class_id"] == POISON_CLASS) &
    (df["timestamp"] >= POISON_START)
].copy()

step = max(1, len(poison_candidates) // len(binary_flag))

bit_index = 0



for idx in poison_candidates.index[::step]:
    if bit_index >= len(binary_flag):
        break

    bit = binary_flag[bit_index]

    if bit == "1":
        df.loc[idx, "f2"] += DELTA
        df.loc[idx, "f5"] -= DELTA
    else:
        df.loc[idx, "f2"] -= DELTA
        df.loc[idx, "f5"] += DELTA

    bit_index += 1



sample = df.sample(SAMPLE_SIZE, random_state=1337)
sample.to_csv(OUTPUT_FILE, index=False)

print("[+] Dataset generated successfully")
print(f"[+] Poisoned class        : {POISON_CLASS}")
print(f"[+] Poison start timestamp: {POISON_START}")
print(f"[+] Flag length (bits)    : {len(binary_flag)}")
print(f"[+] Output file           : {OUTPUT_FILE}")
