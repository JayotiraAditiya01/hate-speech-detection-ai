import pandas as pd

#Load CSV file 
df = pd.read_csv("labeled_data.csv")


df = df[['tweet', 'class']]
df.columns = ['text', 'label']  

# Convert labels
df['label'] = df['label'].apply(lambda x: 1 if x == 0 else 0)

# Show first 5 rows
print("First 5 rows:")
print(df.head())

# Show number of hate vs not hate
print("\n Label Counts (1 = Hate, 0 = Not Hate):")
print(df['label'].value_counts())
