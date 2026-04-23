import pandas as pd
import os
import csv

def clean_csv_data(input_file, output_file=None):
    """
    Clean CSV data by removing timestamp column and empty columns.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional)
    """
    
    # Read the CSV file with error handling
    print(f"Reading {input_file}...")
    
    try:
        # First attempt: normal read
        df = pd.read_csv(input_file)
    except pd.errors.ParserError as e:
        print(f"Parser error encountered: {e}")
        print("Attempting to read with error handling...")
        
        try:
            # Second attempt: skip bad lines
            df = pd.read_csv(input_file, on_bad_lines='skip')
            print("Successfully read file by skipping problematic lines.")
        except Exception as e2:
            print(f"Still having issues: {e2}")
            print("Trying with different parsing options...")
            
            try:
                # Third attempt: more robust reading
                df = pd.read_csv(input_file, 
                               on_bad_lines='skip',
                               sep=',',
                               engine='python',
                               quoting=1,  # QUOTE_ALL
                               skipinitialspace=True)
                print("Successfully read file with robust parsing options.")
            except Exception as e3:
                print(f"Final attempt failed: {e3}")
                print("The CSV file may be severely malformed. Consider manual inspection.")
                return None
    
    # Check if DataFrame was successfully created
    if df is None:
        return None
        
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Show a sample of the data to understand its structure
    print(f"\nFirst few rows:")
    print(df.head(2))
    
    # Remove timestamp column if it exists
    timestamp_columns = ['Timestamp', 'timestamp', 'Time', 'time']
    for col in timestamp_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed column: {col}")
            break
    
    # Remove empty columns (columns that are entirely NaN or empty strings)
    # First, replace empty strings with NaN for easier detection
    df_cleaned = df.replace('', pd.NA)
    
    # Find columns that are entirely empty
    empty_columns = df_cleaned.columns[df_cleaned.isna().all()].tolist()
    
    if empty_columns:
        print(f"Found {len(empty_columns)} empty columns")
        df_cleaned = df_cleaned.drop(columns=empty_columns)
    
    # Also remove columns that have no name (unnamed columns)
    unnamed_columns = [col for col in df_cleaned.columns if 'Unnamed:' in str(col)]
    if unnamed_columns:
        print(f"Found {len(unnamed_columns)} unnamed columns: {unnamed_columns}")
        df_cleaned = df_cleaned.drop(columns=unnamed_columns)
    
    # Remove rows where "Gas name" contains only numbers or is empty
    if 'Gas name' in df_cleaned.columns:
        original_row_count = len(df_cleaned)
        
        # Function to check if a value is a valid gas name (not just numbers)
        def is_valid_gas_name(value):
            if pd.isna(value) or value == '':
                return False
            
            # Convert to string and strip whitespace
            str_value = str(value).strip()
            
            # Check if it's empty after stripping
            if not str_value:
                return False
            
            # Check if it's purely numeric (including floats)
            try:
                float(str_value)
                return False  # It's a number, so not a valid gas name
            except ValueError:
                return True  # Not a number, so it's likely a valid gas name
        
        # Filter out rows with invalid gas names
        df_cleaned = df_cleaned[df_cleaned['Gas name'].apply(is_valid_gas_name)]
        
        removed_rows = original_row_count - len(df_cleaned)
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows with invalid gas names (numbers or empty)")
    
    print(f"Cleaned shape: {df_cleaned.shape}")
    print(f"Remaining columns: {list(df_cleaned.columns)}")
    
    # Show unique gas names to verify filtering worked
    if 'Gas name' in df_cleaned.columns:
        unique_gas_names = df_cleaned['Gas name'].unique()
        print(f"\nUnique gas names found ({len(unique_gas_names)}):")
        for gas_name in sorted(unique_gas_names):
            print(f"  - {gas_name}")
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_cleaned.csv"
    
    # Save the cleaned data
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned data saved to: {output_file}")
    
    return df_cleaned


def inspect_csv_structure(input_file, num_lines=20):
    """
    Inspect the structure of a problematic CSV file
    """
    print(f"Inspecting structure of {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= num_lines:
                break
            fields = line.strip().split(',')
            print(f"Line {i+1}: {len(fields)} fields")
            if i < 3:  # Show first 3 lines content
                print(f"  Content: {line.strip()[:100]}...")
    
    print(f"\nTo manually fix the CSV, you might need to:")
    print("1. Open the file in a text editor")
    print("2. Look for lines with inconsistent number of commas")
    print("3. Remove or fix malformed rows")
    print("4. Save and try the cleaning script again")

# Example usage
if __name__ == "__main__":
    # Replace 'database_robodog_full.csv' with the path to your CSV file
    input_filename = "database_robodog_new.csv"
    
    # Check if file exists
    if os.path.exists(input_filename):
        cleaned_data = clean_csv_data(input_filename)
        
        if cleaned_data is not None:
            # Display first few rows of cleaned data
            print("\nFirst 3 rows of cleaned data:")
            print(cleaned_data.head(3))
            
            # Display summary statistics
            print(f"\nData summary:")
            print(f"Number of rows: {len(cleaned_data)}")
            print(f"Number of columns: {len(cleaned_data.columns)}")
        else:
            print("Failed to clean the data due to parsing errors.")
            print("Running CSV structure inspection...")
            inspect_csv_structure(input_filename)
            
    else:
        print(f"File '{input_filename}' not found. Please update the filename in the script.")
        print("\nTo use this script:")
        print("1. Save your CSV data to a file")
        print("2. Update the 'input_filename' variable with your file path")
        print("3. Run the script")