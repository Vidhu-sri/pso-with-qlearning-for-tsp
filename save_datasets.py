import os
import pandas as pd
from io import StringIO

# Define the path to the directory containing your CSV files
csv_folder_path = r"C:\Users\Lenovo\Desktop\practice+projex\swarmRL\rsets"

# Define the output folder for .tsp files
tsp_folder_path = r'C:\Users\Lenovo\Desktop\practice+projex\swarmRL\pso\tsp-pso\datasets'

# Create the output folder if it doesn't exist
os.makedirs(tsp_folder_path, exist_ok=True)

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]

# Function to convert a DataFrame to TSP format and save it as a .tsp file
def convert_to_tsp_file(df, name, index, output_path):
    tsp_data = f"NAME: {name}_{index}\nTYPE: TSP\nCOMMENT: Generated from CSV\n"
    tsp_data += f"DIMENSION: {df.shape[0]}\nEDGE_WEIGHT_TYPE: EUC_2D\n"
    tsp_data += "NODE_COORD_SECTION\n"
    for _, row in df.iterrows():
        # Skip rows with NaN values in any of the expected columns
        if pd.isna(row['city_id']) or pd.isna(row['x']) or pd.isna(row['y']):
            continue
        # Convert city_id to integer
        city_id = int(row['city_id'])
        x = row['x']
        y = row['y']
        tsp_data += f"{city_id} {x} {y}\n"
    tsp_data += "EOF\n"
    
    # Write the data to a .tsp file
    tsp_file_path = os.path.join(output_path, f"{name}_{chr(97 + index)}.tsp")  # 'a' = chr(97)
    with open(tsp_file_path, 'w') as tsp_file:
        tsp_file.write(tsp_data)
    print(f"Saved {tsp_file_path}")

# Loop over each CSV file, process it, and save as .tsp
for csv_file in csv_files:
    file_path = os.path.join(csv_folder_path, csv_file)
    
    # Read the CSV file, expecting multiple datasets separated by `,,`
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split the content by ',,\n' to separate datasets
    datasets = content.strip().split(',,\n')
    
    # Extract the filename without extension to use as the TSP name
    tsp_name = os.path.splitext(csv_file)[0]
    
    # Process each dataset separately, skipping the header after the first dataset
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        if dataset:  # Ensure the dataset is not empty
            # Add the header back to each dataset after splitting
            if i == 0:
                # Only for the first dataset, the header is already there
                df = pd.read_csv(StringIO(dataset))
            else:
                # Add the header for subsequent datasets
                df = pd.read_csv(StringIO("city_id,x,y\n" + dataset))
            
            # Convert the DataFrame to TSP format and save as a .tsp file
            convert_to_tsp_file(df, tsp_name, i, tsp_folder_path)
