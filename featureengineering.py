import pandas as pd
import numpy as np
import datetime


df = pd.read_csv('cleaned_books_dataset.csv')


df = df.drop_duplicates()

num_cols = [
    'publishing_year', 'book_average_rating', 'book_ratings_count',
    'gross_sales', 'publisher_revenue', 'sale_price', 'units_sold', 'sales_rank'
]
text_cols = ['book_name', 'author', 'author_rating', 'language_code', 'genre', 'publisher']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

for col in text_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()
    df[col] = df[col].replace('nan', 'unknown')

def is_ascii(s):
    return all(ord(c) < 128 for c in str(s))
mask = df[text_cols].applymap(is_ascii).all(axis=1)
df = df[mask].copy()

df = df.dropna(subset=['book_name', 'author'])

df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

df = df[(df['sale_price'] >= 0) & (df['units_sold'] >= 0)]


df = df.reset_index(drop=True)


current_year = datetime.datetime.now().year
df['book_age'] = current_year - df['publishing_year']
df['decade'] = (df['publishing_year'] // 10) * 10


df['average_rating_bin'] = pd.cut(
    df['book_average_rating'],
    bins=[0, 2.5, 4.0, 5],
    labels=['low', 'medium', 'high']
)
df['log_ratings_count'] = np.log1p(df['book_ratings_count'])

#
df['price_category'] = pd.cut(
    df['sale_price'],
    bins=[-1, 200, 500, np.inf],
    labels=['cheap', 'moderate', 'expensive']
)
df['revenue_per_unit'] = df['publisher_revenue'] / df['units_sold'].replace(0, np.nan)
df['revenue_per_unit'] = df['revenue_per_unit'].fillna(0)

df['inv_sales_rank'] = 1 / (df['sales_rank'] + 1)  # inverting, add 1 to prevent div by zero


for col in ['genre', 'publisher', 'language_code']:
    df[f'{col}_code'] = df[col].astype('category').cat.codes


df['author_name_length'] = df['author'].str.len()
author_book_counts = df['author'].value_counts()
df['author_book_count'] = df['author'].map(author_book_counts)


df['book_name_word_count'] = df['book_name'].apply(lambda x: len(str(x).split()))


publisher_book_counts = df['publisher'].value_counts()
df['publisher_book_count'] = df['publisher'].map(publisher_book_counts)
df['publisher_total_units'] = df.groupby('publisher')['units_sold'].transform('sum')


df.to_csv('cleaned_featured_books_dataset.csv', index=False)
print("Complete: cleaned and feature-engineered data saved to cleaned_featured_books_dataset.csv")
