import pandas as pd
import re

df = pd.read_csv('booksdataset.csv')

df = df.drop_duplicates()

num_cols = ['Publishing Year', 'Book_average_rating', 'Book_ratings_count', 'gross sales', 'publisher revenue', 'sale price', 'units sold', 'sales rank']
text_cols = ['Book Name', 'Author', 'Author_Rating', 'language_code', 'genre', 'Publisher']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace('nan', 'unknown')

def is_ascii(s):
    return all(ord(c) < 128 for c in s)
mask = df[text_cols].applymap(is_ascii).all(axis=1)
df = df[mask].copy()

df = df.dropna(subset=['Book Name', 'Author'])

df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

df = df[(df['sale_price'] >= 0) & (df['units_sold'] >= 0)]

df = df.reset_index(drop=True)

df.to_csv('cleaned_books_dataset.csv', index=False)

print("Cleaning complete. Cleaned data saved to cleaned_books_dataset.csv")
