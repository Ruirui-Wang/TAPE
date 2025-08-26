import pandas as pd
import re
import os
import argparse

# Function to extract numerical values, ignoring "± nan"
def extract_value(cell):
    match = re.match(r"([0-9\.]+)", str(cell))
    return float(match.group(1)) if match else None

def process_file(file_path):
    df = pd.read_csv(file_path)
    numeric_cols = df.columns[1:]
    df_cleaned = df.copy()
    df_cleaned[numeric_cols] = df[numeric_cols].applymap(extract_value)
    
    # Compute mean and standard deviation, ignoring NaN values
    summary = df_cleaned.iloc[:4, 1:].agg(['mean', 'std']).round(2)
    summary.loc['summary'] = summary.loc['mean'].astype(str) + ' ± ' + summary.loc['std'].astype(str)
    summary.insert(0, 'Metric', ['summary'] * len(summary))
    
    # Append summary row to the original dataframe
    df_final = pd.concat([df, summary.iloc[-1:].reset_index(drop=True)], ignore_index=True)
    
    # Save the updated CSV file
    output_path = os.path.join(os.path.dirname(file_path), "summary_" + os.path.basename(file_path))
    df_final.to_csv(output_path, index=False)
    print(f"Summary row added and saved to {output_path}")

def process_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            process_file(os.path.join(folder_path, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV files to compute mean and standard deviation.")
    parser.add_argument("path", type=str, help="Path to a CSV file or folder containing CSV files")
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        process_folder(args.path)
    elif os.path.isfile(args.path) and args.path.endswith(".csv"):
        process_file(args.path)
    else:
        print("Invalid path. Please provide a valid CSV file or a folder containing CSV files.")
