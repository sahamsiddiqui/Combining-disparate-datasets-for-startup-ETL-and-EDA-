# Load libraries
import os
import pandas as pd
from datetime import datetime
import country_converter as coco
from rapidfuzz import process, fuzz
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# STEP 1: EXTRACT DATA FROM CSV FILES

# Load the data into pandas DataFrames
try:
    # Get the current directory (for running in terminal)
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data')

    partner_df = pd.read_csv(os.path.join(data_dir, 'partner_data.csv'))
    mapping_df = pd.read_csv(os.path.join(data_dir, 'page_category_mapping.csv'), sep=';')
    conversions_df = pd.read_csv(os.path.join(data_dir, 'reviewfriends_conversions.csv'), sep=';', engine='python')

    print("DataFrames loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. One or more files were not found.")
except pd.errors.EmptyDataError as e:
    print(f"Error: {e}. One or more files are empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# STEP 2: INITIAL EXPLORATION
# Explore partner_df data
print(partner_df['ip_country'].unique())
print(partner_df['important_score'].nunique())
print(partner_df.info())
print(partner_df.shape)  # Output: (61840, 4)
print(partner_df['ip_country'].nunique())  # 211
print(partner_df['country_residency'].nunique())  # 181
print(partner_df['timestamp'].nunique())  # 60935

# Explore conversions_df data
print(conversions_df.shape)  # Output: (102441, 7)
print(conversions_df['id'].nunique())  # 102365
print(conversions_df['session_id'].nunique())  # 86015
print(conversions_df['country_name'].nunique())  # 214
print(conversions_df['created_at'].nunique())  # 100814

# Check for duplicate 'id' in conversions_df
dup_list = conversions_df['id'].value_counts()[conversions_df['id'].value_counts() > 1].index
print(f"Duplicate rows based on 'id':\n{conversions_df[conversions_df['id'].isin(dup_list)]}")

# Remove duplicates
conversions_df = conversions_df[conversions_df['country_name'] != 'West Bank']

# STEP 3: CLEANING AND TRANSFORMATION
# Remove duplicates from all datasets
conversions_df = conversions_df.drop_duplicates()
mapping_df = mapping_df.drop_duplicates().dropna()
partner_df = partner_df.drop_duplicates()

# Convert UNIX datetime to timestamp
partner_df['timestamp'] = pd.to_datetime(partner_df['timestamp'], unit='s')
conversions_df['created_at'] = pd.to_datetime(conversions_df['created_at'])

# Fill NaN values and replace outliers
partner_df['important_score'] = partner_df['important_score'].replace('#REF!', 0, regex=True)
partner_df['ip_country'].fillna('none', inplace=True)
partner_df['ip_country'] = partner_df['ip_country'].replace('-', '', regex=True)
conversions_df['country_name'] = conversions_df['country_name'].replace('_', '', regex=True)

# Remove leading/trailing whitespaces from string values
partner_df['ip_country'] = partner_df['ip_country'].str.strip()
conversions_df['country_name'] = conversions_df['country_name'].str.strip()

# Correct datatypes
partner_df['ip_country'] = partner_df['ip_country'].astype(str)
partner_df['country_residency'] = partner_df['country_residency'].astype(str)
conversions_df['country_name'] = conversions_df['country_name'].astype(str)
conversions_df['measurement_category'] = conversions_df['measurement_category'].astype(str)
conversions_df['ui_element'] = conversions_df['ui_element'].astype(str)

# STEP 4: NORMALIZATION AND FUZZY MATCHING

# Define normalization function
def normalize(text):
    return text.strip().lower()

# Standardize country names using country_converter
def standardize_country_names(df, column_name):
    df[column_name] = df[column_name].str.strip()
    std_countries = coco.convert(names=df[column_name], to='name_short', not_found=None)
    df[column_name] = std_countries
    return df

# Apply standardization
conversions_df = standardize_country_names(conversions_df, 'country_name')
partner_df = standardize_country_names(partner_df, 'ip_country')

# Fuzzy matching function
def fuzzy_match(row, target_col, choices, scorer=fuzz.WRatio, threshold=80):
    match, score, index_fm = process.extractOne(row[target_col], choices, scorer=scorer)
    return match if score > threshold else None

# Fuzzy match for conversions_df
conversions_df['matched_description'] = conversions_df.apply(
    fuzzy_match, target_col='measurement_category',
    choices=mapping_df['measurement_category'].tolist(),
    scorer=fuzz.WRatio,
    threshold=80,
    axis=1
)

# Map matched descriptions
description_to_category = {row['measurement_category']: row['page_category'] for _, row in mapping_df.iterrows()}
conversions_df['page_name'] = conversions_df['matched_description'].map(description_to_category)

# Fuzzy match for partner_df
partner_df['std_country'] = partner_df.apply(
    fuzzy_match, target_col='ip_country',
    choices=conversions_df['country_name'].tolist(),
    scorer=fuzz.WRatio,
    threshold=85,
    axis=1
)

# STEP 5.1: COMBINE DATASETS FOR ANALYSIS
# Prepare columns for merging
conversions_df['created_at'] = pd.to_datetime(conversions_df['created_at'])
partner_df['timestamp'] = pd.to_datetime(partner_df['timestamp'])

# Create composite keys
partner_df['composite_key'] = partner_df['std_country'] + '_' + partner_df['timestamp'].astype(str).str.replace(' ', '').str.replace(':', '')
conversions_df['composite_key'] = conversions_df['country_name'] + '_' + conversions_df['created_at'].astype(str).str.replace(' ', '').str.replace(':', '')

# Remove duplicate composite keys
repeated_keys_conversions = conversions_df["composite_key"].value_counts()[conversions_df["composite_key"].value_counts() > 1].index
conversions_df = conversions_df[~conversions_df["composite_key"].isin(repeated_keys_conversions)]
repeated_keys_partner = partner_df["composite_key"].value_counts()[partner_df["composite_key"].value_counts() > 1].index
partner_df = partner_df[~partner_df["composite_key"].isin(repeated_keys_partner)]

# Merge the two DataFrames on the composite key
composite_merge = pd.merge(conversions_df, partner_df, how='inner', on='composite_key')

# Check the merged result
print(f"Merged dataset shape: {composite_merge.shape}")
print(f"Distinct composite keys in merged dataset: {composite_merge['composite_key'].nunique()}")

# STEP 5.2: Select Required Columns for Analytics

# Select only the necessary columns for analysis
final_columns = [
    'id', 'session_id', 'composite_key', 'std_country', 'country_name', 
    'is_mobile', 'ui_element', 'timestamp', 'created_at', 
    'page_name', 'country_residency', 'important_score'
]
composite_merge = composite_merge[final_columns]

# Ensure important_score is of integer type
composite_merge['important_score'] = composite_merge['important_score'].astype(int)

# STEP 5.3: Feature Engineering for Analytics

# Categorize the `important_score` into bins
bins = [0, 25, 50, 75, 100]
labels = ['Low', 'Medium', 'High', 'Very High']
composite_merge['importance'] = pd.cut(composite_merge['important_score'], bins=bins, labels=labels, include_lowest=True)

# Check the distribution of the 'importance' categories
importance_distribution = composite_merge['importance'].value_counts()
print("\nImportance Distribution:")
print(importance_distribution)

# Convert the 'importance' column to string format before loading into the database
composite_merge['importance'] = composite_merge['importance'].astype(str)

# CHECK THE UNIFIED DATASET FOR INSIGHTS
composite_merge.info()
print(composite_merge.head())

### Step 36: Load (ETL) - Load the transformed data at destination for downstream users

# Function to download the data to a file (either Parquet or CSV)
def load_data_to_file(df, output_file='final_data.parquet', file_format='parquet'):
    if file_format == 'parquet':
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        print(f"Data successfully saved as Parquet to {output_file}")
    elif file_format == 'csv':
        df.to_csv(output_file, index=False)
        print(f"Data successfully saved as CSV to {output_file}")
    else:
        print(f"Unsupported format: {file_format}. Please choose either 'parquet' or 'csv'.")

# Function to load data into a SQLite database (file-based or in-memory)
def load_data_to_sqlite(df, table_name, in_memory=False):
    if in_memory:
        engine = create_engine('sqlite:///:memory:')  # In-memory DB
        print("Using in-memory SQLite database...")
    else:
        engine = create_engine('sqlite:///data.db')  # File-based DB
        print("Using file-based SQLite database (data.db)...")
    
    try:
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print(f"Data successfully loaded into the table '{table_name}' in SQLite database.")
    except Exception as e:
        print(f"Failed to load data into SQLite: {str(e)}")

# Main function that asks the user what they want to do
def process_and_load_data(df):
    print("How would you like to proceed with the data?")
    print("1. Save data to a file (Parquet or CSV)")
    print("2. Upload data to an SQLite database (default)")
    
    user_choice = input("Enter 1 to save to a file or press Enter to upload to SQLite: ").strip()
    
    if user_choice == '1':
        file_format = input("Enter file format (parquet/csv): ").strip().lower()
        output_file = input("Enter the output file name (with extension): ").strip()
        load_data_to_file(df, output_file=output_file, file_format=file_format)
    else:
        in_memory = input("Do you want to use an in-memory database? (yes/no): ").strip().lower() == 'yes'
        table_name = input("Enter the table name for the SQLite database: ").strip()
        load_data_to_sqlite(df, table_name, in_memory=in_memory)

# Call to process and load the data
process_and_load_data(composite_merge)


print( 'the ETL pipeline ran successfully the analytical unified data is now avalable for eda . check the eda pdf for result ') 
