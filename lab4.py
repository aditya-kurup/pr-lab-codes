import math
from collections import Counter
import pandas as pd

def shannon_entropy(data):
    counts = Counter(data)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Show columns and get input
print("Available columns:", df.columns.tolist())
column = input("Enter column name: ")

if column in df.columns:
    print(f"Shannon Entropy of '{column}': {shannon_entropy(df[column])}")
else:
    print(f"Column '{column}' not found.")
