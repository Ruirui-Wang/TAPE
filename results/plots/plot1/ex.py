import pandas as pd 

def load_and_extract_metrics(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Prepare lists to store the extracted data
    model_names = []
    parameters = []
    inference_times = []
    aucs = []
    mrrs = []

    # Iterate over the rows to extract the needed information
    for index, row in df.iterrows():
        model_name = row.iloc[0]
        model_names.append(model_name)
        
        # Extract parameters, inference time, AUC, and MRR directly from the columns
        parameters.append(row['parameters'])
        inference_times.append(row['inference time'])
        aucs.append(row['AUC'])
        mrrs.append(row['MRR'])

    # Create a DataFrame from the extracted data
    extracted_df = pd.DataFrame({
        'Model': model_names,
        'Parameters': parameters,
        'Inference Time': inference_times,
        'AUC': aucs,
        'MRR': mrrs
    })

    return extracted_df

# Example usage:
file_path = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot1/complexity_cora.csv'
extracted_df = load_and_extract_metrics(file_path)
extracted_df.head()  # Display the first few rows of the extracted data