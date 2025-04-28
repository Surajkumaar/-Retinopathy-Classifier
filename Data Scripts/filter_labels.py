import pandas as pd
df = pd.read_csv(r"D:\archive\test.csv")
filtered_df = df[df['level'].isin([4])]
filtered_df.to_csv(r'D:\archive\test_4.csv', index=False)
print(f"Original file had {len(df)} rows")
print(f"Filtered file has {len(filtered_df)} rows")
print("Saved filtered data to 'test_4.csv'")
