import pandas as pd
df = pd.read_csv("C:\Users\User\Downloads\overall_region_corr.csv")

print("Motor column stats:")
print(f"Min: {df.iloc[:, 0].min()}")
print(f"Max: {df.iloc[:, 0].max()}")
print(f"Mean: {df.iloc[:, 0].mean()}")
print(f"First 10 values: {df.iloc[:10, 0].tolist()}")