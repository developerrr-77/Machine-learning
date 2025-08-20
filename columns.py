
import pandas as pd

# Path to the Parquet file
file_path = "/workspaces/Machine-learning/data/transformed/monthly_features.parquet"

# Load the Parquet file
df = pd.read_parquet(file_path)

# Print column names
print("Columns in monthly_features.parquet:")
print(df.columns.tolist())

# Print first 5 rows to inspect data
print("\nFirst 5 rows of the DataFrame:")
print(df.head())
