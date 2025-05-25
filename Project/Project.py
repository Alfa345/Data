# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression # Not directly used for model fitting, statsmodels is preferred
import statsmodels.api as sm
from itertools import combinations
import os # ### ENHANCEMENT: Added for path joining and directory creation

# --- Global variables to store answers for template.csv ---
q_answers = {}

# --- Configuration for Directories ---
### ENHANCEMENT: Define input and output directories
base_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the directory where the script is located
data_sets_dir = os.path.join(base_dir, "data sets")
results_dir = os.path.join(base_dir, "results")

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created directory: {results_dir}")

# --- Matplotlib and Seaborn Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
print("Libraries imported and plotting configured.")

# --- Helper function to save plots ---
def save_plot(filename, tight_layout=True):
    """Saves the current matplotlib plot into the results directory."""
    ### ENHANCEMENT: Save plots into the 'results' directory
    output_filepath = os.path.join(results_dir, filename)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(output_filepath)
    print(f"Plot saved as {output_filepath}")
    plt.close() # Close the plot to free memory

# --- Part 1: Preliminary Analysis (data1.csv) ---
print("\n--- Part 1: Preliminary Analysis (data1.csv) ---")

# Load data1.csv
### ENHANCEMENT: Updated path for data1.csv
data1_path = os.path.join(data_sets_dir, "data1.csv")
try:
    data1 = pd.read_csv(data1_path)
    print(f"Successfully loaded {data1_path} with default comma delimiter.")
except FileNotFoundError:
    print(f"Error: {data1_path} not found. Please ensure it's in the '{data_sets_dir}' directory.")
    exit()
except Exception as e:
    print(f"Error loading {data1_path} with comma delimiter: {e}")
    print("Trying with semicolon delimiter as a fallback...")
    try:
        data1 = pd.read_csv(data1_path, sep=';')
        print(f"Successfully loaded {data1_path} with semicolon delimiter.")
    except Exception as e_semi:
        print(f"Error loading {data1_path} with semicolon delimiter: {e_semi}")
        exit()

# Original column names as per project PDF hints (e.g., 'Minimum_temperature', 'Maximum_temperature', etc.)
# Script's internal expected names: 'City', 'Temp_min_C', 'Temp_max_C', 'Precipitation_mm', 'Ensoleillement_h'

rename_map_data1 = {
    'City': 'City', 
    'Minimum_temperature': 'Temp_min_C',
    'Maximum_temperature': 'Temp_max_C',
    'Rainfall': 'Precipitation_mm',
    'Sunshine_duration': 'Ensoleillement_h'
    # Add other potential original names if necessary:
    # 'Ville': 'City',
    # 'Température minimale (°C)': 'Temp_min_C',
}

data1.columns = data1.columns.str.strip()
data1.rename(columns=rename_map_data1, inplace=True)
print(f"Columns after attempting renaming for data1: {data1.columns.tolist()}")

expected_city_column = 'City'
if expected_city_column in data1.columns:
    data1.set_index(expected_city_column, inplace=True)
    print(f"'{expected_city_column}' set as index for data1.")
else:
    print(f"Error: '{expected_city_column}' column not found in data1.csv after renaming. Actual columns: {data1.columns.tolist()}. Cannot set index.")
    print("Please check the 'rename_map_data1' dictionary and your CSV file's 'City' column name.")
    exit()

expected_processing_cols_data1 = ['Temp_min_C', 'Temp_max_C', 'Precipitation_mm', 'Ensoleillement_h']

if not all(col in data1.columns for col in expected_processing_cols_data1):
    print(f"Warning: Not all expected processing columns found in data1.csv after renaming and index setting.")
    print(f"Expected for processing: {expected_processing_cols_data1}")
    print(f"Actual columns available for processing: {data1.columns.tolist()}")

print("Converting data1 weather columns to numeric...")
for col in expected_processing_cols_data1:
    if col in data1.columns:
        data1[col] = pd.to_numeric(data1[col], errors='coerce')
    else:
        print(f"Warning: Expected column '{col}' not found in data1 for numeric conversion.")
print("Numeric conversion attempt complete for data1.")

print("\n--- Q1: Initial Data and Missing Values (data1) ---")
q_answers['q1a'] = data1.shape[0]
print(f"[q1a] Initial number of cities (rows) in data1: {q_answers['q1a']}")

cities_with_missing_data = data1[data1.isnull().any(axis=1)]
q_answers['q1b'] = cities_with_missing_data.shape[0]
print(f"[q1b] Number of cities affected by missing measurements: {q_answers['q1b']}")

if q_answers['q1b'] > 0:
    print("Cities with missing data (before removal):\n", cities_with_missing_data.index.tolist())
    data1.dropna(inplace=True)
    print(f"Cities with missing data removed. Remaining cities in data1: {data1.shape[0]}")
else:
    print("No cities with missing data found in data1.")

print("\n--- Q2: Extreme Weather Values by City (data1) ---")
weather_vars_q2 = {
    'Temp_min_C': ('lowest minimum temperature', 'highest minimum temperature'),
    'Temp_max_C': ('lowest maximum temperature', 'highest maximum temperature'),
    'Precipitation_mm': ('lowest rainfall', 'highest rainfall'),
    'Ensoleillement_h': ('lowest sunshine duration', 'highest sunshine duration')
}
q_labels_q2 = ['q2a', 'q2b', 'q2c', 'q2d', 'q2e', 'q2f', 'q2g', 'q2h']
label_idx = 0
for var, (desc_low, desc_high) in weather_vars_q2.items():
    q_key_low = q_labels_q2[label_idx]
    q_key_high = q_labels_q2[label_idx+1]

    if var in data1.columns and not data1[var].empty and data1[var].notna().any(): # Check for non-NA values
        min_val = data1[var].min()
        min_city = data1[var].idxmin()
        q_answers[q_key_low] = f"{min_city} ({min_val})"
        print(f"[{q_key_low}] City with {desc_low}: {min_city} ({min_val})")

        max_val = data1[var].max()
        max_city = data1[var].idxmax()
        q_answers[q_key_high] = f"{max_city} ({max_val})"
        print(f"[{q_key_high}] City with {desc_high}: {max_city} ({max_val})")
    else:
        print(f"Warning: Column '{var}' not found, empty, or all NaN in data1 for Q2 analysis. Skipping.")
        q_answers[q_key_low] = "N/A (Column missing, empty, or all NaN)"
        q_answers[q_key_high] = "N/A (Column missing, empty, or all NaN)"
    label_idx += 2
print("Comments Q2: These values represent the extremes in the dataset for each weather variable.")

print("\n--- Q3: Variance of Weather Variables (data1) ---")
numeric_cols_for_variance = data1.select_dtypes(include=np.number).columns.intersection(expected_processing_cols_data1)
if not numeric_cols_for_variance.empty and data1[numeric_cols_for_variance].notna().any().any():
    variances = data1[numeric_cols_for_variance].var()
    print("Calculated variances:")
    var_map_q3 = {
        'Temp_min_C': 'q3a', 'Temp_max_C': 'q3b',
        'Precipitation_mm': 'q3c', 'Ensoleillement_h': 'q3d'
    }
    for var_name, q_id in var_map_q3.items():
        if var_name in variances and pd.notna(variances[var_name]):
            q_answers[q_id] = variances[var_name]
            print(f"[{q_id}] Variance of {var_name}: {q_answers[q_id]:.2f}")
        else:
            q_answers[q_id] = "N/A (Variable not numeric, missing, or variance not calculable)"
            print(f"[{q_id}] Variance of {var_name}: N/A")
else:
    print("No numeric columns with non-NaN values found to calculate variances for Q3.")
    for q_id_suffix in ['a', 'b', 'c', 'd']: q_answers[f'q3{q_id_suffix}'] = "N/A"
    variances = pd.Series(dtype=float) 

if not variances.empty and variances.notna().any():
    var_lowest_variance_name = variances.idxmin()
    var_highest_variance_name = variances.idxmax()

    print(f"\n--- Q4: Stats for '{var_lowest_variance_name}' (Variable with Lowest Variance) ---")
    data_low_var = data1[var_lowest_variance_name]
    q_answers['q4a'] = data_low_var.mean()
    q_answers['q4b'] = data_low_var.median()
    q_answers['q4c'] = data_low_var.std()
    print(f"[q4a] Mean: {q_answers['q4a']:.2f}")
    print(f"[q4b] Median: {q_answers['q4b']:.2f}")
    print(f"[q4c] Standard Deviation: {q_answers['q4c']:.2f}")
    plt.figure()
    data_low_var.hist(bins=15, edgecolor='black')
    plt.title(f"Histogram of {var_lowest_variance_name}")
    plt.xlabel(var_lowest_variance_name)
    plt.ylabel("Frequency")
    save_plot(f"histogram_{var_lowest_variance_name.replace('/', '_').replace(' ', '_')}.png")
    print(f"Comments Q4: The histogram for {var_lowest_variance_name} shows its distribution.")

    print(f"\n--- Q5: Stats for '{var_highest_variance_name}' (Variable with Highest Variance) ---")
    data_high_var = data1[var_highest_variance_name]
    q_answers['q5a'] = data_high_var.mean()
    q_answers['q5b'] = data_high_var.median()
    q_answers['q5c'] = data_high_var.std()
    print(f"[q5a] Mean: {q_answers['q5a']:.2f}")
    print(f"[q5b] Median: {q_answers['q5b']:.2f}")
    print(f"[q5c] Standard Deviation: {q_answers['q5c']:.2f}")
    plt.figure()
    data_high_var.hist(bins=15, edgecolor='black')
    plt.title(f"Histogram of {var_highest_variance_name}")
    plt.xlabel(var_highest_variance_name)
    plt.ylabel("Frequency")
    save_plot(f"histogram_{var_highest_variance_name.replace('/', '_').replace(' ', '_')}.png")
    print(f"Comments Q5: The histogram for {var_highest_variance_name} visualizes its distribution.")
else:
    print("Skipping Q4 & Q5: Variance data is empty or all NaN.")
    for q_id in ['q4a', 'q4b', 'q4c', 'q5a', 'q5b', 'q5c']: q_answers[q_id] = "N/A"

print("\n--- Q6: Linear Correlations Between Weather Variables (data1) ---")
numeric_data_for_corr = data1.select_dtypes(include=np.number).columns.intersection(expected_processing_cols_data1)
if len(numeric_data_for_corr) >= 2 and data1[numeric_data_for_corr].notna().any().any():
    correlation_matrix_vars = data1[numeric_data_for_corr].corr()
    correlations_pairs = correlation_matrix_vars.unstack().sort_values(ascending=False)
    correlations_pairs = correlations_pairs[correlations_pairs.index.get_level_values(0) != correlations_pairs.index.get_level_values(1)]
    correlations_pairs = correlations_pairs.iloc[::2]

    if not correlations_pairs.empty and correlations_pairs.notna().any():
        most_pos_corr_pair = correlations_pairs.index[0]
        most_pos_corr_val = correlations_pairs.iloc[0]
        q_answers['q6a'] = f"{most_pos_corr_pair[0]} & {most_pos_corr_pair[1]} ({most_pos_corr_val:.2f})"
        print(f"[q6a] Two most positively correlated variables: {most_pos_corr_pair[0]} and {most_pos_corr_pair[1]}, Correlation: {most_pos_corr_val:.2f}")

        most_neg_corr_pair = correlations_pairs.index[-1]
        most_neg_corr_val = correlations_pairs.iloc[-1]
        q_answers['q6b'] = f"{most_neg_corr_pair[0]} & {most_neg_corr_pair[1]} ({most_neg_corr_val:.2f})"
        print(f"[q6b] Two most negatively correlated variables: {most_neg_corr_pair[0]} and {most_neg_corr_pair[1]}, Correlation: {most_neg_corr_val:.2f}")

        least_corr_series_abs = correlations_pairs.abs().sort_values()
        least_corr_pair_abs = least_corr_series_abs.index[0]
        original_corr_val_least = correlation_matrix_vars.loc[least_corr_pair_abs[0], least_corr_pair_abs[1]]
        q_answers['q6c'] = f"{least_corr_pair_abs[0]} & {least_corr_pair_abs[1]} ({original_corr_val_least:.2f})"
        print(f"[q6c] Two least correlated variables (closest to 0): {least_corr_pair_abs[0]} and {least_corr_pair_abs[1]}, Correlation: {original_corr_val_least:.2f}")

        vars_to_plot_q6 = [
            (most_pos_corr_pair[0], most_pos_corr_pair[1], "Most_Positive_Correlation"),
            (most_neg_corr_pair[0], most_neg_corr_pair[1], "Most_Negative_Correlation"),
            (least_corr_pair_abs[0], least_corr_pair_abs[1], "Least_Correlation_(closest_to_0)")
        ]

        for var1, var2, title_suffix in vars_to_plot_q6:
            if var1 in data1.columns and var2 in data1.columns: # Ensure columns exist
                plt.figure(figsize=(12, 8))
                sns.scatterplot(x=data1[var1], y=data1[var2])
                for i, city_name in enumerate(data1.index):
                    plt.text(data1[var1].iloc[i], data1[var2].iloc[i], city_name, fontsize=9)
                plt.title(f"Scatter Plot: {var1} vs {var2} ({title_suffix})")
                plt.xlabel(var1)
                plt.ylabel(var2)
                save_plot(f"scatter_{var1.replace('/', '_')}_{var2.replace('/', '_')}.png")
            else:
                print(f"Skipping scatter plot for {var1} vs {var2} as one or both columns are missing.")

        print("Comments Q6: Scatter plots illustrate these correlations.")
    else:
        print("Not enough valid correlation pairs found for Q6.")
        for q_id_suffix in ['a', 'b', 'c']: q_answers[f'q6{q_id_suffix}'] = "N/A"
else:
    print("Skipping Q6: Not enough numeric weather variables or data for correlation analysis.")
    for q_id_suffix in ['a', 'b', 'c']: q_answers[f'q6{q_id_suffix}'] = "N/A"

print("\n--- Q7: Linear Correlations Between Cities (data1) ---")
if len(numeric_data_for_corr) > 0 and data1.shape[0] >=2 and data1[numeric_data_for_corr].notna().any().any() :
    correlation_matrix_cities = data1[numeric_data_for_corr].T.corr() 
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_cities, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix Between Cities (based on weather variables)")
    save_plot("correlation_matrix_cities.png")
    print("Comments Q7: The heatmap shows groups of cities with similar weather patterns.")
    q_answers['q7'] = "Heatmap 'correlation_matrix_cities.png' generated."
else:
    print("Skipping Q7: Not enough data to compute city correlations.")
    q_answers['q7'] = "N/A (Insufficient data)"

print("\n\n--- Part 2: Principal Component Analysis (PCA) ---")
numeric_data_for_pca = data1.select_dtypes(include=np.number)
numeric_data_for_pca = numeric_data_for_pca[numeric_data_for_pca.columns.intersection(expected_processing_cols_data1)]

if numeric_data_for_pca.shape[1] < 2 or not numeric_data_for_pca.notna().any().any():
    print(f"Not enough numeric features or valid data for PCA. Skipping Part 2.")
    q_answers['q8a'] = "N/A (Insufficient features/data)"
    q_answers['q8b'] = "N/A (Insufficient features/data)"
    q_answers['q9'] = "N/A (Insufficient features/data)"
    q_answers['q10'] = "N/A (Insufficient features/data)"
else:
    print(f"Performing PCA on {numeric_data_for_pca.shape[1]} features: {numeric_data_for_pca.columns.tolist()}")
    scaler = StandardScaler()
    data1_scaled_array = scaler.fit_transform(numeric_data_for_pca)
    data1_scaled_df = pd.DataFrame(data1_scaled_array, columns=numeric_data_for_pca.columns, index=numeric_data_for_pca.index)

    print("\n--- Q8: PCA on Weather Data ---")
    pca = PCA(n_components=2)
    principal_components_array = pca.fit_transform(data1_scaled_df)
    pc_df = pd.DataFrame(data=principal_components_array, columns=['PC1', 'PC2'], index=data1_scaled_df.index)

    explained_variance_ratio = pca.explained_variance_ratio_
    q_answers['q8a'] = explained_variance_ratio[0] * 100
    q_answers['q8b'] = explained_variance_ratio[1] * 100
    print(f"[q8a] Percentage of variance explained by PC1: {q_answers['q8a']:.2f}%")
    print(f"[q8b] Percentage of variance explained by PC2: {q_answers['q8b']:.2f}%")

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', data=pc_df, s=50)
    for city_name_idx, row in pc_df.iterrows():
        plt.text(row['PC1'] + 0.05, row['PC2'] + 0.05, city_name_idx, fontsize=9)
    plt.xlabel(f"Principal Component 1 ({q_answers['q8a']:.2f}%)")
    plt.ylabel(f"Principal Component 2 ({q_answers['q8b']:.2f}%)")
    plt.title("PCA of Weather Data (Cities Projected onto PCs)")
    plt.axhline(0, color='grey', lw=0.5, linestyle='--')
    plt.axvline(0, color='grey', lw=0.5, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.7)
    save_plot("pca_cities.png")
    print("Comments Q8: The PCA plot shows cities projected onto the first two principal components.")

    print("\n--- Q9: PCA Correlation Circle ---")
    loadings = pca.components_.T
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, var_name in enumerate(numeric_data_for_pca.columns):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                 head_width=0.05, head_length=0.05, fc='r', ec='r', length_includes_head=True)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var_name, 
                color='r', ha='center', va='center', fontsize=10)
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    ax.add_artist(circle)
    ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel(f"PC1 ({q_answers['q8a']:.2f}%)"); ax.set_ylabel(f"PC2 ({q_answers['q8b']:.2f}%)")
    ax.set_title("PCA Correlation Circle (Variable Contributions to PCs)")
    ax.axhline(0, color='grey', lw=0.5, linestyle='--'); ax.axvline(0, color='grey', lw=0.5, linestyle='--')
    ax.grid(True, linestyle=':', alpha=0.7); ax.set_aspect('equal', adjustable='box')
    save_plot("correlation_circle.png")
    q_answers['q9'] = "Plot 'correlation_circle.png' generated."
    print("Comments Q9: The correlation circle shows how original variables contribute to the principal components.")

    print("\n--- Q10: PCA Biplot (Scores and Loadings) ---")
    fig, ax1 = plt.subplots(figsize=(14, 10))
    ax1.scatter(pc_df['PC1'], pc_df['PC2'], c='blue', label='Cities', alpha=0.7, s=50)
    for city_name_idx, row in pc_df.iterrows():
        ax1.text(row['PC1'] + 0.05, row['PC2'] + 0.05, city_name_idx, fontsize=9, c='blue')
    ax1.set_xlabel(f"Principal Component 1 ({q_answers['q8a']:.2f}%) - City Scores")
    ax1.set_ylabel(f"Principal Component 2 ({q_answers['q8b']:.2f}%) - City Scores", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(0, color='grey', lw=0.5, linestyle='--'); ax1.axvline(0, color='grey', lw=0.5, linestyle='--')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax2 = ax1.twinx().twiny()
    for i, var_name in enumerate(numeric_data_for_pca.columns):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], 
                  head_width=0.03, head_length=0.03, fc='red', ec='red', alpha=0.8, length_includes_head=True)
        ax2.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var_name, 
                 color='red', ha='center', va='center', fontsize=10, alpha=0.8)
    ax2.set_xlim([-1.2, 1.2]); ax2.set_ylim([-1.2, 1.2])
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.title("PCA Biplot: City Scores (Blue) and Variable Loadings (Red)")
    save_plot("pca_biplot.png")
    q_answers['q10'] = "Plot 'pca_biplot.png' generated."
    print("Comments Q10: The biplot superimposes city locations and variable contributions.")

# --- Part 3: Simple Linear Regression (data2.csv) ---
print("\n\n--- Part 3: Simple Linear Regression (data2.csv) ---")
### ENHANCEMENT: Updated path for data2.csv
data2_path = os.path.join(data_sets_dir, "data2.csv")
try:
    data2 = pd.read_csv(data2_path)
    print(f"Successfully loaded {data2_path} with default comma delimiter.")
except FileNotFoundError:
    print(f"Error: {data2_path} not found. Please ensure it's in the '{data_sets_dir}' directory.")
    exit()
except Exception as e:
    print(f"Error loading {data2_path} with comma delimiter: {e}")
    print(f"Trying with semicolon delimiter for {data2_path}...")
    try:
        data2 = pd.read_csv(data2_path, sep=';')
        print(f"Successfully loaded {data2_path} with semicolon delimiter.")
    except Exception as e_semi_d2:
        print(f"Error loading {data2_path} with semicolon delimiter: {e_semi_d2}")
        exit()

rename_map_data2 = {
    'Maximum_temperature': 'Max_Temperature_Paris'
    # 'MOIS': 'Month', # Uncomment and adjust if your 'Month' column is named 'MOIS'
    # 'ANNEE': 'Year',  # Uncomment and adjust if your 'Year' column is named 'ANNEE'
}
if rename_map_data2:
    data2.columns = data2.columns.str.strip()
    data2.rename(columns=rename_map_data2, inplace=True)
    print(f"Columns after attempting renaming for data2: {data2.columns.tolist()}")
else:
    print("No rename map for data2 specified; assuming columns are 'Month', 'Year', 'Max_Temperature_Paris'.")

required_cols_data2 = ['Month', 'Year', 'Max_Temperature_Paris']
if not all(col in data2.columns for col in required_cols_data2):
    print(f"Error: Not all required columns {required_cols_data2} found in data2.csv.")
    print(f"Actual columns: {data2.columns.tolist()}")
    # exit()

if 'Max_Temperature_Paris' in data2.columns:
    data2['Max_Temperature_Paris'] = pd.to_numeric(data2['Max_Temperature_Paris'], errors='coerce')
    print("'Max_Temperature_Paris' column converted to numeric.")
else:
    print("Warning: 'Max_Temperature_Paris' column not found in data2.csv.")

month_order_english = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
data2['month_ID'] = np.nan

if 'Month' in data2.columns:
    print(f"Processing 'Month' column from data2 (dtype: {data2['Month'].dtype}).")
    if pd.api.types.is_numeric_dtype(data2['Month']):
        data2['month_ID'] = data2['Month'] - 1
        print("Numeric 'Month' column converted to 0-11 'month_ID'.")
    else:
        # month_name_map_to_english = {'janvier': 'January', ...} # Customize if needed
        # current_month_series = data2['Month'].str.lower().str.strip().map(month_name_map_to_english)
        current_month_series = data2['Month'].astype(str).str.lower().str.strip()
        month_to_id_map = {name.lower(): i for i, name in enumerate(month_order_english)}
        data2['month_ID'] = current_month_series.map(month_to_id_map)
        if data2['month_ID'].isnull().any():
            print("Warning: Some 'Month' values in data2.csv could not be mapped to 'month_ID'.")
        else:
            print("String 'Month' column successfully mapped to 'month_ID'.")
else:
    print("Warning: 'Month' column not found in data2.csv.")

if 'Year' in data2.columns:
    data2['Year'] = pd.to_numeric(data2['Year'], errors='coerce')

data2_paris_2024 = data2[
    (data2['Year'] == 2024) &
    pd.notna(data2['month_ID']) &
    pd.notna(data2['Max_Temperature_Paris'])
].copy()
data2_paris_2024.sort_values('month_ID', inplace=True)
data2_paris_2024.reset_index(drop=True, inplace=True)
print(f"Filtered data for Paris 2024: {len(data2_paris_2024)} valid monthly records found.")

print("\n--- Q11: Temperature Evolution in Paris (2024) ---")
if not data2_paris_2024.empty and 'Max_Temperature_Paris' in data2_paris_2024.columns and 'month_ID' in data2_paris_2024.columns and data2_paris_2024['month_ID'].notna().any():
    plt.figure(figsize=(10,6))
    plt.plot(data2_paris_2024['month_ID'], data2_paris_2024['Max_Temperature_Paris'], marker='o', linestyle='-')
    plt.xticks(ticks=range(12), labels=month_order_english, rotation=45, ha="right")
    plt.xlabel("Month"); plt.ylabel("Maximum Temperature (°C) in Paris (2024)")
    plt.title("Evolution of Maximum Temperature in Paris (2024)")
    save_plot("paris_temp_2024.png")
    q_answers['q11'] = "Plot 'paris_temp_2024.png' generated."
    print("Comments Q11: Temperature typically shows a seasonal pattern.")
else:
    print("Skipping Q11 plot: Not enough valid data for Paris 2024.")
    q_answers['q11'] = "N/A (Insufficient data)"

print("\n--- Q12: Optimal Simple Linear Regression for Paris 2024 Temperatures ---")
best_n_slr = -1; best_adj_r2_slr = -float('inf')
best_model_slr_results = None
optimal_b0_slr, optimal_b1_slr, optimal_r2_slr = None, None, None

if data2_paris_2024.empty or len(data2_paris_2024) < 2 or \
   'month_ID' not in data2_paris_2024.columns or \
   'Max_Temperature_Paris' not in data2_paris_2024.columns or \
   not data2_paris_2024['month_ID'].notna().any() or \
   not data2_paris_2024['Max_Temperature_Paris'].notna().any() :
    print("Not enough valid data for Paris 2024 for SLR. Skipping Q12-Q14.")
    for q_id_suffix in ['a', 'b', 'c', 'd', 'e']: q_answers[f'q12{q_id_suffix}'] = "N/A"
    for q_id_suffix in ['a', 'b']: q_answers[f'q13{q_id_suffix}'] = "N/A"; q_answers[f'q14{q_id_suffix}'] = "N/A"
else:
    print(f"Finding optimal 'n' for SLR using up to {len(data2_paris_2024)} months of 2024 data.")
    for n_months_slr in range(1, min(12, len(data2_paris_2024)) + 1):
        if n_months_slr < 2: continue
        current_data_slr = data2_paris_2024.tail(n_months_slr)
        if current_data_slr['month_ID'].nunique() < 2 and n_months_slr > 1 : # Avoid perfect multicollinearity if month_ID is constant for n > 1
             if current_data_slr['month_ID'].var() == 0: continue
        X_slr = current_data_slr['month_ID']; y_slr = current_data_slr['Max_Temperature_Paris']
        X_slr_sm = sm.add_constant(X_slr)
        try:
            model_slr = sm.OLS(y_slr, X_slr_sm).fit()
            current_adj_r2 = model_slr.rsquared_adj
            if current_adj_r2 > best_adj_r2_slr:
                best_adj_r2_slr = current_adj_r2; best_n_slr = n_months_slr
                best_model_slr_results = model_slr
                optimal_b0_slr = model_slr.params['const']
                optimal_b1_slr = model_slr.params['month_ID']
                optimal_r2_slr = model_slr.rsquared
        except Exception as e_ols_slr:
            # print(f"Error OLS n_months_slr={n_months_slr}: {e_ols_slr}")
            pass # Continue if a particular n fails

    if best_model_slr_results:
        q_answers['q12a'] = best_n_slr; q_answers['q12b'] = best_adj_r2_slr
        q_answers['q12c'] = optimal_r2_slr; q_answers['q12d'] = optimal_b0_slr
        q_answers['q12e'] = optimal_b1_slr
        print(f"[q12a] Optimal n: {q_answers['q12a']}")
        print(f"[q12b] Adj. R2: {q_answers['q12b']:.4f}")
        print(f"[q12c] R2: {q_answers['q12c']:.4f}")
        print(f"[q12d] Beta0: {q_answers['q12d']:.4f}")
        print(f"[q12e] Beta1: {q_answers['q12e']:.4f}")
        optimal_data_slr = data2_paris_2024.tail(best_n_slr)
        plt.figure(figsize=(10,6))
        plt.scatter(optimal_data_slr['month_ID'], optimal_data_slr['Max_Temperature_Paris'], 
                    label=f'Actual Data (last {best_n_slr} months of 2024)', color='blue')
        pred_y_slr = best_model_slr_results.predict(sm.add_constant(optimal_data_slr['month_ID']))
        plt.plot(optimal_data_slr['month_ID'], pred_y_slr, color='red', 
                 label=f'Optimal SLR (n={best_n_slr}, Adj. R2={best_adj_r2_slr:.3f})')
        relevant_month_ids = optimal_data_slr['month_ID'].unique()
        relevant_month_names = [month_order_english[int(m_id)] for m_id in sorted(relevant_month_ids) if 0 <= int(m_id) < 12] # Ensure m_id is valid index
        plt.xticks(ticks=sorted(relevant_month_ids), labels=relevant_month_names, rotation=45, ha="right")
        plt.xlabel("Month ID"); plt.ylabel("Max Temp (°C)")
        plt.title(f"Optimal SLR for Paris 2024 (Last {best_n_slr} months)")
        plt.legend(); save_plot(f"optimal_simple_regression_n{best_n_slr}.png")
        print(f"Comments Q12: Optimal n={best_n_slr} months yields adj. R2={best_adj_r2_slr:.4f}.")
    else:
        print("Could not determine optimal SLR model.")
        for q_id_suffix in ['a', 'b', 'c', 'd', 'e']: q_answers[f'q12{q_id_suffix}'] = "N/A"

    print("\n--- Q13: Prediction for January 2025 (SLR) ---")
    if best_model_slr_results:
        jan_2025_month_id = 0 
        X_pred_jan_slr = pd.DataFrame({'const': [1], 'month_ID': [jan_2025_month_id]})
        predicted_temp_jan_2025_slr = best_model_slr_results.predict(X_pred_jan_slr)[0]
        q_answers['q13a'] = predicted_temp_jan_2025_slr
        print(f"[q13a] Predicted temp Jan 2025 (SLR): {q_answers['q13a']:.2f}°C")
        actual_temp_jan_2025_pdf = 7.5
        difference_slr = predicted_temp_jan_2025_slr - actual_temp_jan_2025_pdf
        q_answers['q13b'] = difference_slr
        print(f"[q13b] Difference (SLR predicted - actual) Jan 2025: {q_answers['q13b']:.2f}°C")
    else:
        print("No optimal SLR model for Q13 predictions."); q_answers['q13a'] = "N/A"; q_answers['q13b'] = "N/A"

    print("\n--- Q14: Hypothesis Test for Beta1 (SLR) ---")
    if best_model_slr_results:
        if 'month_ID' in best_model_slr_results.pvalues:
            p_value_beta1_slr = best_model_slr_results.pvalues['month_ID']
            q_answers['q14a'] = p_value_beta1_slr
            print(f"[q14a] P-value for Beta1 (SLR): {q_answers['q14a']:.4f}")
            alpha_slr = 0.05; is_significant_slr = p_value_beta1_slr < alpha_slr
            q_answers['q14b'] = "Yes" if is_significant_slr else "No"
            print(f"[q14b] Significant linear relationship (alpha=5%)? {'Yes' if is_significant_slr else 'No'}")
        else:
            print("Warning: 'month_ID' not in p-values of SLR model."); q_answers['q14a'] = "N/A"; q_answers['q14b'] = "N/A"
    else:
        print("No optimal SLR model for Q14 hypothesis testing."); q_answers['q14a'] = "N/A"; q_answers['q14b'] = "N/A"

# --- Part 4: Multivariate Linear Regression ---
print("\n\n--- Part 4: Multivariate Linear Regression ---")
data2_paris_2023 = data2[
    (data2['Year'] == 2023) & pd.notna(data2['month_ID']) & pd.notna(data2['Max_Temperature_Paris'])
].copy()
data2_paris_2023.sort_values('month_ID', inplace=True); data2_paris_2023.reset_index(drop=True, inplace=True)
print(f"Filtered data for Paris 2023: {len(data2_paris_2023)} valid records.")

print("\n--- Q15: Temperature Evolution (2023 vs 2024) ---")
if not data2_paris_2023.empty and not data2_paris_2024.empty and \
   all(col in data2_paris_2023 for col in ['month_ID', 'Max_Temperature_Paris']) and \
   all(col in data2_paris_2024 for col in ['month_ID', 'Max_Temperature_Paris']) and \
   data2_paris_2023['month_ID'].notna().any() and data2_paris_2024['month_ID'].notna().any():
    plt.figure(figsize=(12,7))
    plt.plot(data2_paris_2023['month_ID'], data2_paris_2023['Max_Temperature_Paris'], marker='o', linestyle='-', label='Paris 2023 Max Temp')
    plt.plot(data2_paris_2024['month_ID'], data2_paris_2024['Max_Temperature_Paris'], marker='x', linestyle='--', label='Paris 2024 Max Temp')
    plt.xticks(ticks=range(12), labels=month_order_english, rotation=45, ha="right")
    plt.xlabel("Month"); plt.ylabel("Max Temp (°C)")
    plt.title("Max Temp in Paris (2023 vs 2024)")
    plt.legend(); save_plot("paris_temp_2023_2024.png")
    q_answers['q15'] = "Plot 'paris_temp_2023_2024.png' generated."
    print("Comments Q15: Both years show seasonal trends.")
else:
    print("Skipping Q15 plot: Insufficient data."); q_answers['q15'] = "N/A"

if not data2_paris_2023.empty and not data2_paris_2024.empty:
    full_temp_data_for_lags = pd.concat([
        data2_paris_2023[['Year', 'month_ID', 'Max_Temperature_Paris']],
        data2_paris_2024[['Year', 'month_ID', 'Max_Temperature_Paris']]
    ]).sort_values(['Year', 'month_ID']).reset_index(drop=True)
    print(f"Combined 2023-2024 data for lags: {len(full_temp_data_for_lags)} records.")
else:
    print("Cannot create combined 2023-2024 data for MLR."); full_temp_data_for_lags = pd.DataFrame()

if not full_temp_data_for_lags.empty:
    mlr_df_with_lags = full_temp_data_for_lags.copy()
    print("Creating lagged temperature features (T_lag_1 to T_lag_12)...")
    for lag in range(1, 13):
        mlr_df_with_lags[f'T_lag_{lag}'] = mlr_df_with_lags['Max_Temperature_Paris'].shift(lag)
    mlr_train_df = mlr_df_with_lags[mlr_df_with_lags['Year'] == 2024].dropna()
    print(f"MLR training data (2024 with complete lags): {len(mlr_train_df)} records.")
else:
    mlr_train_df = pd.DataFrame()

if mlr_train_df.empty or mlr_train_df.shape[0] < 2:
    print("Not enough data for MLR. Skipping Q16-Q17.")
    for q_id_suffix in ['a', 'b']: q_answers[f'q16{q_id_suffix}'] = "N/A"; q_answers['q17'] = "N/A"
else:
    print("\n--- Q16: Optimal Multivariate Linear Regression (MLR) for Paris 2024 ---")
    num_lag_variables_max = 12
    q_answers['q16a'] = 2**num_lag_variables_max - 1
    print(f"[q16a] Possible combinations of up to 12 lags: {q_answers['q16a']}")
    lag_column_names = [f'T_lag_{i}' for i in range(1, num_lag_variables_max + 1)]
    available_lag_cols_for_mlr = [col for col in lag_column_names if col in mlr_train_df.columns and not mlr_train_df[col].isnull().all()]
    
    if not available_lag_cols_for_mlr:
        print("No valid lag features for MLR. Skipping Q16 model fitting.")
        q_answers['q16b'] = "N/A"; q_answers['q17'] = "N/A"
    else:
        print(f"Finding optimal MLR using {len(available_lag_cols_for_mlr)} available lags: {available_lag_cols_for_mlr}")
        y_mlr_train = mlr_train_df['Max_Temperature_Paris']
        best_adj_r2_mlr = -float('inf'); best_combo_mlr_vars = None
        best_model_mlr_results = None
        for k_predictors in range(1, len(available_lag_cols_for_mlr) + 1):
            for combo_vars in combinations(available_lag_cols_for_mlr, k_predictors):
                X_mlr_combo_df = mlr_train_df[list(combo_vars)]
                X_mlr_sm_combo_df = sm.add_constant(X_mlr_combo_df)
                if X_mlr_sm_combo_df.shape[0] <= X_mlr_sm_combo_df.shape[1]: continue
                try:
                    model_mlr_iter = sm.OLS(y_mlr_train, X_mlr_sm_combo_df).fit()
                    if model_mlr_iter.rsquared_adj > best_adj_r2_mlr:
                        best_adj_r2_mlr = model_mlr_iter.rsquared_adj
                        best_combo_mlr_vars = list(combo_vars)
                        best_model_mlr_results = model_mlr_iter
                except Exception as e_ols_mlr_iter: pass
        
        if best_model_mlr_results:
            q_answers['q16b'] = len(best_combo_mlr_vars)
            print(f"[q16b] Selected variables in optimal MLR: {q_answers['q16b']}")
            print(f"Optimal MLR combination: {best_combo_mlr_vars}")
            print(f"Optimal MLR adj. R2: {best_adj_r2_mlr:.4f}")
            print("Optimal MLR parameters:\n", best_model_mlr_results.params)
            f_pvalue_mlr = best_model_mlr_results.f_pvalue; alpha_mlr_f_test = 0.05
            mlr_overall_significant = f_pvalue_mlr < alpha_mlr_f_test
            print(f"P-value F-statistic MLR: {f_pvalue_mlr:.4f}")
            print(f"Overall linear relationship (F-test alpha=5%)? {'Yes' if mlr_overall_significant else 'No'}")
        else:
            print("Could not determine optimal MLR model."); q_answers['q16b'] = "N/A"; q_answers['q17'] = "N/A"

    print("\n--- Q17: Predictions using Optimal MLR Model ---")
    if best_model_mlr_results and best_combo_mlr_vars and not full_temp_data_for_lags.empty:
        actual_temp_jan_2025_pdf_q17 = 7.5
        if len(full_temp_data_for_lags) >= 12:
            recent_12_actual_temps = full_temp_data_for_lags['Max_Temperature_Paris'].tail(12).values
            features_for_jan_2025_pred = {}; valid_lags_for_jan_pred = True
            for lag_col_name in best_combo_mlr_vars:
                lag_number = int(lag_col_name.split('_')[-1])
                if (12 - lag_number) >= 0 and (12 - lag_number) < len(recent_12_actual_temps):
                    features_for_jan_2025_pred[lag_col_name] = recent_12_actual_temps[12 - lag_number]
                else: valid_lags_for_jan_pred = False; break
            
            if valid_lags_for_jan_pred:
                features_jan_2025_df = pd.DataFrame([features_for_jan_2025_pred])
                features_jan_2025_df_sm = sm.add_constant(features_jan_2025_df[best_combo_mlr_vars], has_constant='add')
                predicted_temp_jan_2025_mlr = best_model_mlr_results.predict(features_jan_2025_df_sm)[0]
                diff_jan_2025_mlr = predicted_temp_jan_2025_mlr - actual_temp_jan_2025_pdf_q17
                q_answers['q17'] = diff_jan_2025_mlr
                print(f"Predicted temp Jan 2025 (MLR): {predicted_temp_jan_2025_mlr:.2f}°C")
                print(f"[q17] Difference (MLR predicted - actual) Jan 2025: {q_answers['q17']:.2f}°C")

                print("\nPredicting Feb, Mar, Apr 2025 (MLR using actuals):")
                temps_history_for_future = list(recent_12_actual_temps)
                temps_history_for_future.append(actual_temp_jan_2025_pdf_q17)
                future_months_actuals_pdf = {'Feb 2025': 8.6, 'Mar 2025': 14.6, 'Apr 2025': 20.0}
                for month_year_str, actual_future_temp_pdf in future_months_actuals_pdf.items():
                    if len(temps_history_for_future) < 12: break
                    current_lags_source = temps_history_for_future[-12:]
                    features_for_future_month = {}; valid_lags_for_future = True
                    for lag_col_name in best_combo_mlr_vars:
                        lag_number = int(lag_col_name.split('_')[-1])
                        if (12 - lag_number) >= 0 and (12 - lag_number) < len(current_lags_source):
                            features_for_future_month[lag_col_name] = current_lags_source[12 - lag_number]
                        else: valid_lags_for_future = False; break
                    if not valid_lags_for_future: break
                    features_future_df = pd.DataFrame([features_for_future_month])
                    features_future_df_sm = sm.add_constant(features_future_df[best_combo_mlr_vars], has_constant='add')
                    predicted_temp_future_mlr = best_model_mlr_results.predict(features_future_df_sm)[0]
                    diff_future_mlr = predicted_temp_future_mlr - actual_future_temp_pdf
                    print(f"  {month_year_str}: Pred(MLR)={predicted_temp_future_mlr:.2f}°C, Actual={actual_future_temp_pdf:.1f}°C, Diff={diff_future_mlr:.2f}°C")
                    temps_history_for_future.append(actual_future_temp_pdf)
            else: q_answers['q17'] = "N/A (Missing lag data Jan 2025 MLR)"
        else: q_answers['q17'] = "N/A (Insufficient historical data for lags)"
    elif 'q17' not in q_answers : q_answers['q17'] = "N/A (No Q16 MLR model or data)"


# --- Generate content for template.csv ---
print("\n\n--- Generating Answers for Legrandjacques_Slama_Sartorius.csv ---")
template_csv_lines = ["Question_ID,Answer"]
question_structure_for_csv = {
    1: ['a', 'b'], 2: [chr(ord('a') + i) for i in range(8)], 3: ['a', 'b', 'c', 'd'],
    4: ['a', 'b', 'c'], 5: ['a', 'b', 'c'], 6: ['a', 'b', 'c'], 7: [], 8: ['a', 'b'],
    9: [], 10: [], 11: [], 12: ['a', 'b', 'c', 'd', 'e'], 13: ['a', 'b'], 14: ['a', 'b'],
    15: [], 16: ['a', 'b'], 17: []
}
for q_num in range(1, 18):
    parts = question_structure_for_csv.get(q_num, [])
    if not parts:
        q_key = f"q{q_num}"
        answer = q_answers.get(q_key, 'Not Computed')
        answer_str = f"{answer:.4f}" if isinstance(answer, float) else str(answer)
        template_csv_lines.append(f"{q_key},{answer_str}")
    else:
        for part_letter in parts:
            q_key = f"q{q_num}{part_letter}"
            answer = q_answers.get(q_key, 'Not Computed')
            answer_str = f"{answer:.4f}" if isinstance(answer, float) else str(answer)
            template_csv_lines.append(f"{q_key},{answer_str}")

# --- Generate content for template.csv ---
# ... (code to build template_csv_content_final) ...

template_csv_content_final = "\n".join(template_csv_lines)
print("\n--- Content for answers CSV ---")
print(template_csv_content_final)

### ENHANCEMENT: Updated output CSV filename and path
output_csv_filename_final = os.path.join(results_dir, "Legrandjacques_Slama_Sartorius.csv")

# ADD THIS LINE FOR DEBUGGING:
print(f"DEBUG: Attempting to save CSV to: {output_csv_filename_final}")

try:
    with open(output_csv_filename_final, "w", encoding='utf-8') as f:
        f.write(template_csv_content_final)
    print(f"\nAnswers saved to {output_csv_filename_final}")
# ... rest of the script
except Exception as e_write_csv:
    print(f"\nError writing {output_csv_filename_final}: {e_write_csv}")

print("\n--- Python script execution finished. ---")
print(f"Review all generated .png plot images in '{results_dir}' and the console output.")
print(f"Ensure all [qij] values in your report are filled from the console output or '{output_csv_filename_final}'.")