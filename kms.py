import pickle
import pandas as pd

# Load pickle file
with open("pickle/combined_data_labeled.pkl", "rb") as f:
    data = pickle.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Filter rows with non-null labels
df = df.loc[df["label"].notna()]

# Drop unnecessary column
df = df.drop(columns=["is_feedback"])

# Count the number of each unique label
label_counts = df["label"].value_counts()
print(label_counts)