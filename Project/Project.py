# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import combinations

# --- Global variables to store answers for template.csv ---
q_answers = {}

# --- Matplotlib and Seaborn Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# --- Helper function to save plots ---
def save_plot(filename, tight_layout=True):
    """Saves the current matplotlib plot."""
    if tight_layout:
        plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close() # Close the plot to free memory

# --- Part 1: Preliminary Analysis (data1.csv) ---
print("--- Part 1: Preliminary Analysis (data1.csv) ---")

# Load data1.csv
try:
    # Try with comma delimiter first, as indicated by user's error log
    data1 = pd.read_csv("data1.csv") 
except FileNotFoundError:
    print("Error: data1.csv not found. Please place it in the same directory as the script.")
    exit()
except Exception as e: # Catch other potential parsing errors
    print(f"Error loading data1.csv with comma delimiter: {e}")
    print("Trying with semicolon delimiter as a fallback...")
    try:
        data1 = pd.read_csv("data1.csv", sep=';') 
    except Exception as e_semi:
        print(f"Error loading data1.csv with semicolon delimiter: {e_semi}")
        exit()

# Original column names from user's CSV seem to be:
# 'City', 'Minimum_temperature', 'Maximum_temperature', 'Rainfall', 'Sunshine_duration'

# Standardize column names to what the script expects internally
# Script's internal expected names: 'Ville' (now 'City'), 'Temp_min_C', 'Temp_max_C', 'Precipitation_mm', 'Ensoleillement_h'

rename_map_data1 = {
    'City': 'City', # This will be used as index directly
    'Minimum_temperature': 'Temp_min_C',
    'Maximum_temperature': 'Temp_max_C',
    'Rainfall': 'Precipitation_mm',
    'Sunshine_duration': 'Ensoleillement_h'
}

# Clean current column names (e.g. remove spaces if any, though user's log implies they are clean)
data1.columns = data1.columns.str.strip()
data1.rename(columns=rename_map_data1, inplace=True)


# Set 'City' as index (changed from 'Ville')
expected_city_column = 'City' # Updated from 'Ville'
if expected_city_column in data1.columns:
    data1.set_index(expected_city_column, inplace=True)
else:
    print(f"Error: '{expected_city_column}' column not found in data1.csv after renaming. Actual columns: {data1.columns.tolist()}. Cannot set index.")
    exit()

# Script's internal expected column names for processing (after index is set)
expected_processing_cols_data1 = ['Temp_min_C', 'Temp_max_C', 'Precipitation_mm', 'Ensoleillement_h']

# Verify that the expected processing columns are present
if not all(col in data1.columns for col in expected_processing_cols_data1):
    print(f"Warning: Not all expected processing columns found in data1.csv after renaming.")
    print(f"Expected for processing: {expected_processing_cols_data1}")
    print(f"Actual columns found: {data1.columns.tolist()}")
    # You might need to adjust rename_map_data1 if there are further discrepancies

# Convert data to numeric, coercing errors to NaN
for col in data1.columns: # Iterate over remaining columns (weather variables)
    data1[col] = pd.to_numeric(data1[col], errors='coerce')


# Q1: Number of cities and missing data
print("\n--- Q1 ---")
q_answers['q1a'] = data1.shape[0]
print(f"[q1a] Initial number of cities: {q_answers['q1a']}")

cities_with_missing_data = data1[data1.isnull().any(axis=1)]
q_answers['q1b'] = cities_with_missing_data.shape[0]
print(f"[q1b] Number of cities affected by missing measurements: {q_answers['q1b']}")

if q_answers['q1b'] > 0:
    print("Cities with missing data:\n", cities_with_missing_data)
    data1.dropna(inplace=True)
    print(f"Cities with missing data removed. Remaining cities: {data1.shape[0]}")
else:
    print("No cities with missing data.")

# Q2: Cities associated with min/max values
print("\n--- Q2 ---")
# These variable names should match the keys in rename_map_data1 values (the standardized names)
weather_vars_q2 = {
    'Temp_min_C': ('lowest minimum temperature', 'highest minimum temperature'),
    'Temp_max_C': ('lowest maximum temperature', 'highest maximum temperature'),
    'Precipitation_mm': ('lowest rainfall', 'highest rainfall'),
    'Ensoleillement_h': ('lowest sunshine duration', 'highest sunshine duration')
}
q_labels_q2 = ['q2a', 'q2b', 'q2c', 'q2d', 'q2e', 'q2f', 'q2g', 'q2h']
idx = 0
for var, (desc_low, desc_high) in weather_vars_q2.items():
    if var not in data1.columns:
        print(f"Warning: Column {var} not found for Q2 analysis. Skipping.")
        q_answers[q_labels_q2[idx]] = "N/A (Column missing)"
        q_answers[q_labels_q2[idx+1]] = "N/A (Column missing)"
        idx += 2
        continue

    min_val = data1[var].min()
    min_city = data1[var].idxmin()
    q_answers[q_labels_q2[idx]] = f"{min_city} ({min_val})"
    print(f"[{q_labels_q2[idx]}] City with {desc_low}: {min_city} ({min_val})")
    idx += 1

    max_val = data1[var].max()
    max_city = data1[var].idxmax()
    q_answers[q_labels_q2[idx]] = f"{max_city} ({max_val})"
    print(f"[{q_labels_q2[idx]}] City with {desc_high}: {max_city} ({max_val})")
    idx += 1
print("Comments Q2: These values represent the extremes in the dataset for each weather variable, highlighting regional climatic differences across France.")

# Q3: Variance
print("\n--- Q3 ---")
variances = data1.var() # Uses standardized column names
if 'Temp_min_C' in variances:
    q_answers['q3a'] = variances['Temp_min_C']
    print(f"[q3a] Variance of minimum temperature: {q_answers['q3a']:.2f}")
else: q_answers['q3a'] = "N/A"

if 'Temp_max_C' in variances:
    q_answers['q3b'] = variances['Temp_max_C']
    print(f"[q3b] Variance of maximum temperature: {q_answers['q3b']:.2f}")
else: q_answers['q3b'] = "N/A"

if 'Precipitation_mm' in variances:
    q_answers['q3c'] = variances['Precipitation_mm']
    print(f"[q3c] Variance of total rainfall: {q_answers['q3c']:.2f}")
else: q_answers['q3c'] = "N/A"

if 'Ensoleillement_h' in variances:
    q_answers['q3d'] = variances['Ensoleillement_h']
    print(f"[q3d] Variance of sunshine duration: {q_answers['q3d']:.2f}")
else: q_answers['q3d'] = "N/A"

print("Comments Q3: Variance indicates the spread of data. Rainfall and Sunshine duration show higher variability compared to temperatures.")

# Q4 & Q5: Stats for variable with lowest/highest variance
if not variances.empty:
    var_lowest_variance_name = variances.idxmin() # This will be a standardized name
    var_highest_variance_name = variances.idxmax() # This will be a standardized name

    print(f"\n--- Q4: Stats for {var_lowest_variance_name} (lowest variance) ---")
    data_low_var = data1[var_lowest_variance_name]
    q_answers['q4a'] = data_low_var.mean()
    q_answers['q4b'] = data_low_var.median()
    q_answers['q4c'] = data_low_var.std()
    print(f"[q4a] Mean: {q_answers['q4a']:.2f}")
    print(f"[q4b] Median: {q_answers['q4b']:.2f}")
    print(f"[q4c] Standard Deviation: {q_answers['q4c']:.2f}")
    data_low_var.hist(bins=15)
    plt.title(f"Histogram of {var_lowest_variance_name}")
    plt.xlabel(var_lowest_variance_name)
    plt.ylabel("Frequency")
    save_plot(f"histogram_{var_lowest_variance_name.replace('/', '_')}.png") # Sanitize filename
    print(f"Comments Q4: The histogram for {var_lowest_variance_name} shows its distribution. Compare mean and median for skewness.")

    print(f"\n--- Q5: Stats for {var_highest_variance_name} (highest variance) ---")
    data_high_var = data1[var_highest_variance_name]
    q_answers['q5a'] = data_high_var.mean()
    q_answers['q5b'] = data_high_var.median()
    q_answers['q5c'] = data_high_var.std()
    print(f"[q5a] Mean: {q_answers['q5a']:.2f}")
    print(f"[q5b] Median: {q_answers['q5b']:.2f}")
    print(f"[q5c] Standard Deviation: {q_answers['q5c']:.2f}")
    data_high_var.hist(bins=15)
    plt.title(f"Histogram of {var_highest_variance_name}")
    plt.xlabel(var_highest_variance_name)
    plt.ylabel("Frequency")
    save_plot(f"histogram_{var_highest_variance_name.replace('/', '_')}.png") # Sanitize filename
    print(f"Comments Q5: The histogram for {var_highest_variance_name} shows its distribution. Note the wider spread due to higher variance.")
else:
    print("Skipping Q4 & Q5 due to lack of variance data (possibly all columns were non-numeric or missing).")
    for q_id in ['q4a', 'q4b', 'q4c', 'q5a', 'q5b', 'q5c']: q_answers[q_id] = "N/A"


# Q6: Linear correlations between weather variables
print("\n--- Q6 ---")
correlation_matrix_vars = data1.corr() # Uses standardized column names
# Unstack, sort, and remove duplicates/self-correlations
correlations_pairs = correlation_matrix_vars.unstack().sort_values(ascending=False)
correlations_pairs = correlations_pairs[correlations_pairs.index.get_level_values(0) != correlations_pairs.index.get_level_values(1)] # remove self-correlation
# To remove duplicates (varA, varB) and (varB, varA)
correlations_pairs = correlations_pairs.iloc[::2] # Take every second one after sorting

if not correlations_pairs.empty:
    most_pos_corr_idx = correlations_pairs.index[0]
    most_pos_corr_val = correlations_pairs.iloc[0]
    q_answers['q6a'] = f"{most_pos_corr_idx[0]} & {most_pos_corr_idx[1]} ({most_pos_corr_val:.2f})"
    print(f"[q6a] Two most positively correlated variables: {most_pos_corr_idx[0]} and {most_pos_corr_idx[1]}, Correlation: {most_pos_corr_val:.2f}")

    most_neg_corr_idx = correlations_pairs.index[-1]
    most_neg_corr_val = correlations_pairs.iloc[-1]
    q_answers['q6b'] = f"{most_neg_corr_idx[0]} & {most_neg_corr_idx[1]} ({most_neg_corr_val:.2f})"
    print(f"[q6b] Two most negatively correlated variables: {most_neg_corr_idx[0]} and {most_neg_corr_idx[1]}, Correlation: {most_neg_corr_val:.2f}")

    # Least correlated (closest to zero)
    least_corr_series = correlations_pairs.abs().sort_values()
    least_corr_idx = least_corr_series.index[0]
    # Get original signed value from the original correlation matrix using the identified pair
    original_corr_val_least = correlation_matrix_vars.loc[least_corr_idx[0], least_corr_idx[1]]
    q_answers['q6c'] = f"{least_corr_idx[0]} & {least_corr_idx[1]} ({original_corr_val_least:.2f})"
    print(f"[q6c] Two least correlated variables: {least_corr_idx[0]} and {least_corr_idx[1]}, Correlation: {original_corr_val_least:.2f}")

    # Scatter plots for Q6
    vars_to_plot = [
        (most_pos_corr_idx[0], most_pos_corr_idx[1], "Most_Positive_Correlation"),
        (most_neg_corr_idx[0], most_neg_corr_idx[1], "Most_Negative_Correlation"),
        (least_corr_idx[0], least_corr_idx[1], "Least_Correlation")
    ]

    for var1, var2, title_suffix in vars_to_plot:
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=data1[var1], y=data1[var2])
        for i, city_name_idx in enumerate(data1.index): # city_name_idx is used as 'city' was set as index
            plt.text(data1[var1].iloc[i], data1[var2].iloc[i], city_name_idx, fontsize=9)
        plt.title(f"Scatter Plot: {var1} vs {var2} ({title_suffix})")
        plt.xlabel(var1)
        plt.ylabel(var2)
        save_plot(f"scatter_{var1.replace('/', '_')}_{var2.replace('/', '_')}.png") # Sanitize filenames
    print("Comments Q6: Scatter plots illustrate these correlations. For example, min and max temperatures are highly positively correlated as expected.")
else:
    print("Skipping Q6 due to issues with correlation matrix (e.g. not enough numeric data).")
    for q_id in ['q6a', 'q6b', 'q6c']: q_answers[q_id] = "N/A"


# Q7: Linear correlations between cities
print("\n--- Q7 ---")
# Transpose data so cities are rows and variables are columns, then correlate cities
correlation_matrix_cities = data1.T.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_cities, annot=False, cmap='coolwarm', fmt=".1f") # Annot may be too cluttered
plt.title("Correlation Matrix Between Cities")
save_plot("correlation_matrix_cities.png")
print("Comments Q7: The heatmap shows groups of cities with similar weather patterns. Dark red indicates high positive correlation, dark blue high negative.")


# --- Part 2: Principal Component Analysis (PCA) ---
print("\n\n--- Part 2: Principal Component Analysis (PCA) ---")
# Ensure data1 has numeric columns for PCA (already converted, but good to check shape)
numeric_data_for_pca = data1.select_dtypes(include=np.number)
if numeric_data_for_pca.shape[1] < 2 : # Need at least 2 features for PCA
    print(f"Not enough numeric features for PCA (found {numeric_data_for_pca.shape[1]}). Skipping Part 2.")
    q_answers['q8a'] = "N/A"
    q_answers['q8b'] = "N/A"
else:
    # Center and reduce data
    scaler = StandardScaler()
    # Use only numeric columns for scaling and PCA
    data1_scaled = scaler.fit_transform(numeric_data_for_pca)
    data1_scaled_df = pd.DataFrame(data1_scaled, columns=numeric_data_for_pca.columns, index=numeric_data_for_pca.index)

    # Q8: Apply PCA
    print("\n--- Q8 ---")
    pca = PCA(n_components=2) # We are interested in the first two principal components
    principal_components = pca.fit_transform(data1_scaled_df)
    pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=data1_scaled_df.index)

    explained_variance_ratio = pca.explained_variance_ratio_
    q_answers['q8a'] = explained_variance_ratio[0] * 100
    q_answers['q8b'] = explained_variance_ratio[1] * 100
    print(f"[q8a] Percentage of variance explained by PC1: {q_answers['q8a']:.2f}%")
    print(f"[q8b] Percentage of variance explained by PC2: {q_answers['q8b']:.2f}%")

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', data=pc_df)
    for i, city_name_idx in enumerate(pc_df.index):
        plt.text(pc_df['PC1'].iloc[i], pc_df['PC2'].iloc[i], city_name_idx, fontsize=9)
    plt.xlabel(f"Principal Component 1 ({q_answers['q8a']:.2f}%)")
    plt.ylabel(f"Principal Component 2 ({q_answers['q8b']:.2f}%)")
    plt.title("PCA of Weather Data (Cities)")
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid(True)
    save_plot("pca_cities.png")
    print("Comments Q8: The PCA plot shows cities projected onto the first two principal components. Cities close together have similar weather profiles according to these components.")

    # Q9: Correlation Circle
    print("\n--- Q9 ---")
    loadings = pca.components_.T # Rows are original features (numeric_data_for_pca.columns), columns are PCs
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # Important: Use numeric_data_for_pca.columns for variable names here
    for i, var_name in enumerate(numeric_data_for_pca.columns):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, head_length=0.05, fc='r', ec='r')
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var_name, color='r', ha='center', va='center')

    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    ax.add_artist(circle)

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel(f"PC1 ({q_answers['q8a']:.2f}%)")
    ax.set_ylabel(f"PC2 ({q_answers['q8b']:.2f}%)")
    ax.set_title("Correlation Circle")
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    ax.grid(True)
    save_plot("correlation_circle.png")
    print("Comments Q9: The correlation circle shows how original variables contribute to the principal components. Variables close to each other are positively correlated. Variables opposite are negatively correlated. Variables near the circle edge are well represented by these PCs.")

    # Q10: Superimpose PCA results and correlation circle
    print("\n--- Q10 ---")
    fig, ax1 = plt.subplots(figsize=(14, 10))

    ax1.scatter(pc_df['PC1'], pc_df['PC2'], c='blue', label='Cities')
    for i, city_name_idx in enumerate(pc_df.index):
        ax1.text(pc_df['PC1'].iloc[i], pc_df['PC2'].iloc[i], city_name_idx, fontsize=9, c='blue')
    ax1.set_xlabel(f"Principal Component 1 ({q_answers['q8a']:.2f}%)")
    ax1.set_ylabel(f"Principal Component 2 ({q_answers['q8b']:.2f}%)", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(0, color='grey', lw=0.5)
    ax1.axvline(0, color='grey', lw=0.5)
    ax1.grid(True)

    ax2 = ax1.twinx().twiny() 
    # Important: Use numeric_data_for_pca.columns for variable names here
    for i, var_name in enumerate(numeric_data_for_pca.columns):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.03, head_length=0.03, fc='red', ec='red', alpha=0.7)
        ax2.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, var_name, color='red', ha='center', va='center', alpha=0.7)
    
    # Determine max absolute loading values for scaling plot limits if needed
    max_abs_loading_pc1 = np.abs(loadings[:,0]).max() if loadings.shape[0] > 0 else 1.0
    max_abs_loading_pc2 = np.abs(loadings[:,1]).max() if loadings.shape[0] > 0 else 1.0
    ax2.set_xlim([-1.2 * max_abs_loading_pc1, 1.2 * max_abs_loading_pc1])
    ax2.set_ylim([-1.2 * max_abs_loading_pc2, 1.2 * max_abs_loading_pc2])
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.title("Biplot: PCA Scores (Cities) and Loadings (Variables)")
    save_plot("pca_biplot.png")
    print("Comments Q10: The biplot superimposes cities and variable contributions. Cities are influenced by variables pointing towards them. For Q2, cities at the extremes of the plot, aligned with specific variable arrows, likely correspond to min/max values for those variables. E.g., a city far along an axis strongly correlated with 'Temp_max_C' might be the one with the highest max temperature.")


# --- Part 3: Simple Linear Regression (data2.csv) ---
print("\n\n--- Part 3: Simple Linear Regression (data2.csv) ---")
try:
    # Assuming data2.csv is more standard, try comma first, then semicolon
    data2 = pd.read_csv("data2.csv") 
except FileNotFoundError:
    print("Error: data2.csv not found. Please place it in the same directory as the script.")
    exit()
except Exception as e:
    print(f"Error loading data2.csv with comma delimiter: {e}")
    print("Trying with semicolon delimiter for data2.csv...")
    try:
        data2 = pd.read_csv("data2.csv", sep=';')
    except Exception as e_comma:
        print(f"Error loading data2.csv with semicolon delimiter: {e_comma}")
        exit()

# Standardize column names for data2.csv if necessary.
# Project PDF implies: 'Month', 'Year', 'Max_Temperature_Paris'
# Let's assume these are the direct column names in data2.csv or can be mapped.
# If they are different, a rename_map_data2 would be needed here similar to data1.
# For example:
# rename_map_data2 = {
# 'MOIS': 'Month', 'ANNEE': 'Year', 'TEMP_MAX_PARIS': 'Max_Temperature_Paris'
# }
# data2.rename(columns=rename_map_data2, inplace=True)

# Ensure 'Month', 'Year', 'Max_Temperature_Paris' columns exist
required_cols_data2 = ['Month', 'Year', 'Max_Temperature_Paris']
if not all(col in data2.columns for col in required_cols_data2):
    print(f"Error: Not all required columns {required_cols_data2} found in data2.csv.")
    print(f"Actual columns: {data2.columns.tolist()}")
    # Consider exiting or handling this error if critical columns are missing
    # For now, we'll proceed, but downstream code might fail.
    # exit() # Uncomment if these columns are absolutely necessary

# Ensure 'Month' is categorical for plotting order if it's string names
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'Max_Temperature_Paris' to numeric
if 'Max_Temperature_Paris' in data2.columns:
    data2['Max_Temperature_Paris'] = pd.to_numeric(data2['Max_Temperature_Paris'], errors='coerce')
else:
    print("Warning: 'Max_Temperature_Paris' column not found in data2.csv. Regression will fail.")

if 'Month' in data2.columns and data2['Month'].dtype == 'object':
    try:
        # Attempt to map month names to English if they are in French or other languages
        # This is a common source of error if month names don't match 'month_order'
        # Example: {'janvier': 'January', 'février': 'February', ...}
        # For simplicity, we assume English names or direct mapping to month_ID will work.
        data2['Month'] = pd.Categorical(data2['Month'], categories=month_order, ordered=True)
    except Exception as e_month_cat:
        print(f"Warning: Could not set 'Month' as categorical with predefined order: {e_month_cat}. Plotting order might be incorrect if months are strings.")
        pass 

# Create month_ID (0 for Jan to 11 for Dec)
if 'Month' in data2.columns:
    if pd.api.types.is_numeric_dtype(data2['Month']):
        data2['month_ID'] = data2['Month'] - 1 # Assuming 1-12
    else: # If 'Month' is string name or categorical
        month_map = {name: i for i, name in enumerate(month_order)}
        # If 'Month' is already categorical with the right categories, this map might not be needed
        # but it's safer to ensure month_ID is 0-11
        if isinstance(data2['Month'].dtype, pd.CategoricalDtype):
             # Map from the string representation of the category
            data2['month_ID'] = data2['Month'].astype(str).map(month_map)
        else: # General object type
            data2['month_ID'] = data2['Month'].map(month_map)
else:
    print("Warning: 'Month' column not found in data2.csv. 'month_ID' cannot be created. Regression will likely fail.")
    data2['month_ID'] = pd.Series(dtype='int') # Create an empty series to prevent NameError later, but it's problematic


# Filter for Paris 2024 data, sort, and handle potential NaNs in month_ID or Temp
data2_paris_2024 = data2[(data2['Year'] == 2024) & pd.notna(data2['month_ID']) & pd.notna(data2['Max_Temperature_Paris'])]
data2_paris_2024 = data2_paris_2024.sort_values('month_ID').reset_index(drop=True)


# Q11: Evolution of temperature in 2024
print("\n--- Q11 ---")
if not data2_paris_2024.empty and 'Max_Temperature_Paris' in data2_paris_2024.columns and 'month_ID' in data2_paris_2024.columns:
    plt.figure(figsize=(10,6))
    plt.plot(data2_paris_2024['month_ID'], data2_paris_2024['Max_Temperature_Paris'], marker='o', linestyle='-')
    plt.xticks(ticks=range(12), labels=month_order) # Ensure all 12 months are labeled
    plt.xlabel("Month")
    plt.ylabel("Maximum Temperature (°C) in Paris (2024)")
    plt.title("Evolution of Maximum Temperature in Paris (2024)")
    save_plot("paris_temp_2024.png")
    print("Comments Q11: Temperature shows a seasonal pattern, peaking in summer months and lower in winter.")
else:
    print("Skipping Q11 plot: Not enough valid data for Paris 2024 temperatures or month IDs.")

# Q12: Optimal n for linear regression
print("\n--- Q12 ---")
best_n = -1
best_r2_adj = -float('inf')
best_model_results = None # Stores the statsmodels OLSResults object
optimal_b0, optimal_b1, optimal_r2 = None, None, None # Store parameters of the best model

# Ensure data2_paris_2024 has the required columns and enough data
if data2_paris_2024.empty or len(data2_paris_2024) < 2 or \
   'month_ID' not in data2_paris_2024.columns or \
   'Max_Temperature_Paris' not in data2_paris_2024.columns:
    print("Not enough valid data for Paris 2024 to perform simple linear regression. Skipping Q12-Q14.")
    for q_id in ['q12a', 'q12b', 'q12c', 'q12d', 'q12e', 'q13a', 'q13b', 'q14a', 'q14b']:
        q_answers[q_id] = "N/A"
else:
    for n_months in range(1, 13): 
        if n_months > len(data2_paris_2024): continue 
        
        current_data = data2_paris_2024.tail(n_months)
        if len(current_data) < 2: continue # Need at least 2 points for OLS regression

        X = current_data['month_ID']
        y = current_data['Max_Temperature_Paris']
        X_sm = sm.add_constant(X) 

        try:
            model = sm.OLS(y, X_sm).fit()
            
            # Check if model has enough observations relative to parameters for R_adj
            if model.nobs > model.df_model + 1: 
                r2_adj = model.rsquared_adj
                if r2_adj > best_r2_adj:
                    best_r2_adj = r2_adj
                    best_n = n_months
                    best_model_results = model # Save the whole model object
                    optimal_b0 = model.params['const']
                    optimal_b1 = model.params['month_ID'] # Ensure 'month_ID' is the name used in X_sm
                    optimal_r2 = model.rsquared
            # Fallback for cases where n_months is small (e.g., n=1, model.nobs = 1)
            # For n=1 (1 data point), OLS is not meaningful.
            # For n=2 (2 data points), R2 is 1, R2_adj might be calculable or 1.
            # The loop starts at n_months=1, current_data.tail(1) -> 1 row
            # sm.OLS needs at least 2 observations for a meaningful regression with intercept and slope.
            # The condition `len(current_data) < 2` above handles this.
        except Exception as e_ols:
            print(f"Error during OLS for n_months={n_months}: {e_ols}")
            continue


    if best_model_results: # Check if a best model was found
        q_answers['q12a'] = best_n
        q_answers['q12b'] = best_r2_adj
        q_answers['q12c'] = optimal_r2 
        q_answers['q12d'] = optimal_b0
        q_answers['q12e'] = optimal_b1

        print(f"[q12a] Optimal value of n: {q_answers['q12a']}")
        print(f"[q12b] Associated adjusted R-squared (R2_adj): {q_answers['q12b']:.4f}")
        print(f"[q12c] Associated R-squared (R2): {q_answers['q12c']:.4f}")
        print(f"[q12d] Beta0 (intercept): {q_answers['q12d']:.4f}")
        print(f"[q12e] Beta1 (slope for month_ID): {q_answers['q12e']:.4f}")

        optimal_data = data2_paris_2024.tail(best_n)
        plt.figure(figsize=(10,6))
        plt.scatter(optimal_data['month_ID'], optimal_data['Max_Temperature_Paris'], label=f'Actual Data (last {best_n} months)')
        # Predict using the best model's parameters on the data it was trained on
        pred_y = best_model_results.predict(sm.add_constant(optimal_data['month_ID']))
        plt.plot(optimal_data['month_ID'], pred_y, color='red', label=f'Optimal Linear Regression (n={best_n})')
        
        # Use full month range for x-ticks for context, even if model uses fewer
        plt.xticks(ticks=range(12), labels=month_order[:12]) 
        plt.xlabel("Month ID (0=Jan, ..., 11=Dec)")
        plt.ylabel("Maximum Temperature (°C)")
        plt.title(f"Optimal Simple Linear Regression for Paris 2024 (n={best_n})")
        plt.legend()
        save_plot(f"optimal_simple_regression_n{best_n}.png")
        print("Comments Q12: The optimal n and regression parameters are found. The plot shows the fit.")
    else:
        print("Could not determine optimal simple linear regression model (e.g. not enough data or all fits were poor).")
        for q_id in ['q12a', 'q12b', 'q12c', 'q12d', 'q12e']: q_answers[q_id] = "N/A"


    # Q13: Prediction for Jan 2025
    print("\n--- Q13 ---")
    if best_model_results: # Use the stored model object
        jan_2025_month_id = 0 
        # Create a DataFrame for prediction, matching the structure of X_sm
        X_pred_jan = pd.DataFrame({'const': [1], 'month_ID': [jan_2025_month_id]})
        predicted_temp_jan_2025 = best_model_results.predict(X_pred_jan)[0]
        
        q_answers['q13a'] = predicted_temp_jan_2025
        print(f"[q13a] Predicted temperature for January 2025: {q_answers['q13a']:.2f}°C")
        
        actual_temp_jan_2025 = 7.5
        difference = predicted_temp_jan_2025 - actual_temp_jan_2025
        q_answers['q13b'] = difference
        print(f"[q13b] Difference between predicted and actual temperature: {q_answers['q13b']:.2f}°C")
    else:
        print("No optimal model from Q12 to make predictions for Q13.")
        q_answers['q13a'] = "N/A"
        q_answers['q13b'] = "N/A"

    # Q14: Null slope hypothesis for Beta1
    print("\n--- Q14 ---")
    if best_model_results: # Use the stored model object
        # Ensure 'month_ID' was indeed a regressor in the model
        if 'month_ID' in best_model_results.pvalues:
            p_value_beta1 = best_model_results.pvalues['month_ID']
            q_answers['q14a'] = p_value_beta1
            print(f"[q14a] P-value for Beta1: {q_answers['q14a']:.4f}")
            
            alpha = 0.05
            is_significant = p_value_beta1 < alpha
            q_answers['q14b'] = "Yes" if is_significant else "No"
            print(f"[q14b] Is there a linear relationship (alpha=5%)? {'Yes' if is_significant else 'No'}")
        else:
            print("Warning: 'month_ID' not found in p-values of the best model. Cannot perform hypothesis test for Beta1.")
            q_answers['q14a'] = "N/A (Regressor not found)"
            q_answers['q14b'] = "N/A"
    else:
        print("No optimal model from Q12 for hypothesis testing in Q14.")
        q_answers['q14a'] = "N/A"
        q_answers['q14b'] = "N/A"


# --- Part 4: Multivariate Linear Regression (data2.csv) ---
print("\n\n--- Part 4: Multivariate Linear Regression ---")
# Prepare data for 2023, ensuring valid month_ID and Temperature
data2_paris_2023 = data2[(data2['Year'] == 2023) & pd.notna(data2['month_ID']) & pd.notna(data2['Max_Temperature_Paris'])]
data2_paris_2023 = data2_paris_2023.sort_values('month_ID')


# Q15: Superimpose temperatures for 2023 and 2024
print("\n--- Q15 ---")
if not data2_paris_2023.empty and not data2_paris_2024.empty and \
   all(col in data2_paris_2023.columns for col in ['month_ID', 'Max_Temperature_Paris']) and \
   all(col in data2_paris_2024.columns for col in ['month_ID', 'Max_Temperature_Paris']):
    plt.figure(figsize=(12,7))
    plt.plot(data2_paris_2023['month_ID'], data2_paris_2023['Max_Temperature_Paris'], marker='o', linestyle='-', label='Paris 2023')
    plt.plot(data2_paris_2024['month_ID'], data2_paris_2024['Max_Temperature_Paris'], marker='x', linestyle='--', label='Paris 2024')
    plt.xticks(ticks=range(12), labels=month_order)
    plt.xlabel("Month")
    plt.ylabel("Maximum Temperature (°C)")
    plt.title("Evolution of Maximum Temperature in Paris (2023 vs 2024)")
    plt.legend()
    save_plot("paris_temp_2023_2024.png")
    print("Comments Q15: Both years show similar seasonal trends. Differences might exist in specific months.")
else:
    print("Skipping Q15 plot: Not enough valid data for Paris 2023 or 2024.")


# Prepare data for multivariate regression
# Combine 2023 and 2024 data, ensure sorted by year then month_ID
# Use already cleaned and validated data2_paris_2023 and data2_paris_2024
if not data2_paris_2023.empty and not data2_paris_2024.empty:
    full_temp_data = pd.concat([
        data2_paris_2023[['Year', 'month_ID', 'Max_Temperature_Paris']],
        data2_paris_2024[['Year', 'month_ID', 'Max_Temperature_Paris']]
    ]).sort_values(['Year', 'month_ID']).reset_index(drop=True)
else:
    print("Cannot create full_temp_data for multivariate regression due to missing 2023 or 2024 data.")
    full_temp_data = pd.DataFrame() # Empty DataFrame to avoid errors later

# Create lagged features
if not full_temp_data.empty:
    mlr_df = full_temp_data.copy()
    for lag in range(1, 13): # Lags from 1 to 12 months
        mlr_df[f'T_lag_{lag}'] = mlr_df['Max_Temperature_Paris'].shift(lag)
    
    # We are predicting for months in 2024.
    mlr_train_df = mlr_df[mlr_df['Year'] == 2024].dropna() # Drop rows with NaNs from lagging
else:
    mlr_train_df = pd.DataFrame() # Empty DataFrame

if mlr_train_df.empty or mlr_train_df.shape[0] < 2: 
    print("Not enough data to build multivariate regression model after lagging. Skipping Q16-Q17.")
    for q_id in ['q16a', 'q16b', 'q17']:
        q_answers[q_id] = "N/A"
else:
    # Q16: Combinations of variables
    print("\n--- Q16 ---")
    num_lag_variables = 12 # Max possible lags
    q_answers['q16a'] = 2**num_lag_variables - 1 
    print(f"[q16a] Number of possible combinations of up to 12 variables: {q_answers['q16a']}")

    lag_columns = [f'T_lag_{i}' for i in range(1, 13)]
    available_lag_columns = [col for col in lag_columns if col in mlr_train_df.columns]
    
    if not available_lag_columns:
        print("No lag features available in mlr_train_df for multivariate regression. Skipping Q16 model fitting.")
        q_answers['q16b'] = "N/A (No features)"
        # Set other Q16/Q17 answers to N/A
        q_answers['q17'] = "N/A"

    else:
        y_mlr = mlr_train_df['Max_Temperature_Paris']
        best_adj_r2_mlr = -float('inf')
        best_combo_mlr = None
        best_model_mlr_results = None # Stores statsmodels OLSResults object

        for k_predictors in range(1, len(available_lag_columns) + 1):
            for combo in combinations(available_lag_columns, k_predictors):
                X_mlr_combo = mlr_train_df[list(combo)]
                X_mlr_sm_combo = sm.add_constant(X_mlr_combo)
                
                if X_mlr_sm_combo.shape[0] <= X_mlr_sm_combo.shape[1]:
                    continue 

                try:
                    model_mlr = sm.OLS(y_mlr, X_mlr_sm_combo).fit()
                    if model_mlr.nobs > model_mlr.df_model + 1: 
                        if model_mlr.rsquared_adj > best_adj_r2_mlr:
                            best_adj_r2_mlr = model_mlr.rsquared_adj
                            best_combo_mlr = list(combo)
                            best_model_mlr_results = model_mlr # Store the model object
                except Exception as e_ols_mlr:
                    print(f"Error during MLR OLS for combo {combo}: {e_ols_mlr}")
                    continue
        
        if best_model_mlr_results: # Check if a model was successfully fitted
            q_answers['q16b'] = len(best_combo_mlr)
            print(f"[q16b] Number of selected variables in optimal combination: {q_answers['q16b']}")
            print(f"Optimal adjusted R2: {best_adj_r2_mlr:.4f}")
            print(f"Selected variables: {best_combo_mlr}")
            print("Associated parameters (coefficients):")
            print(best_model_mlr_results.params)

            f_pvalue = best_model_mlr_results.f_pvalue
            alpha_mlr = 0.05
            mlr_significant = f_pvalue < alpha_mlr
            print(f"P-value for F-statistic: {f_pvalue:.4f}")
            print(f"Is there a linear relationship between selected variables and target (alpha=5%)? {'Yes' if mlr_significant else 'No'}")
            
            q_answers['q16_adj_r2'] = best_adj_r2_mlr
            q_answers['q16_selected_vars'] = ", ".join(best_combo_mlr)
            q_answers['q16_params'] = best_model_mlr_results.params.to_string()
            q_answers['q16_f_pvalue'] = f_pvalue
            q_answers['q16_overall_significant'] = "Yes" if mlr_significant else "No"

        else:
            print("Could not determine optimal multivariate regression model.")
            q_answers['q16b'] = "N/A"
            q_answers['q17'] = "N/A" # If no model for Q16, then no prediction for Q17


    # Q17: Prediction for Jan 2025 and future months
    print("\n--- Q17 ---")
    if best_model_mlr_results and best_combo_mlr and not full_temp_data.empty:
        # Get the last 12 known temperatures (up to Dec 2024) from full_temp_data
        # Ensure full_temp_data has enough rows for .tail(12)
        if len(full_temp_data) >= 12:
            last_12_temps = full_temp_data['Max_Temperature_Paris'].tail(12).values
            
            features_jan_2025 = {}
            for lag_col in best_combo_mlr: 
                lag_num = int(lag_col.split('_')[-1]) 
                if (12 - lag_num) >= 0 and (12 - lag_num) < len(last_12_temps):
                     features_jan_2025[lag_col] = last_12_temps[12 - lag_num]
                else: 
                    features_jan_2025[lag_col] = np.nan 

            features_jan_2025_df = pd.DataFrame([features_jan_2025])
            # Check for NaNs introduced if some lags couldn't be found
            if features_jan_2025_df[best_combo_mlr].isnull().any().any():
                print("Could not form complete feature vector for Jan 2025 MLR prediction due to missing lag data from historicals.")
                q_answers['q17'] = "N/A (Missing lag data for Jan 2025 prediction)"
            else:
                features_jan_2025_df_sm = sm.add_constant(features_jan_2025_df[best_combo_mlr], has_constant='add')
                # Ensure columns are in the same order as training
                # The constant term is named 'const' by sm.add_constant
                # The model was trained with 'const' and then the combo variables.
                cols_for_pred = ['const'] + best_combo_mlr
                features_jan_2025_df_sm = features_jan_2025_df_sm[cols_for_pred]

                predicted_temp_jan_2025_mlr = best_model_mlr_results.predict(features_jan_2025_df_sm)[0]
                actual_temp_jan_2025_val = 7.5
                diff_jan_2025_mlr = predicted_temp_jan_2025_mlr - actual_temp_jan_2025_val
                q_answers['q17'] = diff_jan_2025_mlr
                print(f"Predicted temperature for January 2025 (Multivariate): {predicted_temp_jan_2025_mlr:.2f}°C")
                print(f"[q17] Difference for Jan 2025: {q_answers['q17']:.2f}°C")
                print("Comment Q17 (Jan): Compare this difference with the simple linear regression model's error.")

                # Predicting Feb, Mar, Apr 2025
                print("\nPredicting for Feb, Mar, Apr 2025 (Multivariate):")
                temps_for_future_lags = list(last_12_temps) 
                temps_for_future_lags.append(actual_temp_jan_2025_val) # Add actual Jan 2025
                
                future_months_actuals = {'Feb': 8.6, 'Mar': 14.6, 'Apr': 20.0}
                
                for month_name, actual_future_temp in future_months_actuals.items():
                    if len(temps_for_future_lags) < 12:
                        print(f"Not enough historical data ({len(temps_for_future_lags)} months) to form 12 lags for {month_name} prediction.")
                        break
                    
                    recent_temps_for_lags = temps_for_future_lags[-12:]
                    features_future = {}
                    missing_a_lag = False
                    for lag_col in best_combo_mlr:
                        lag_num = int(lag_col.split('_')[-1])
                        if (12 - lag_num) >= 0 and (12 - lag_num) < len(recent_temps_for_lags):
                            features_future[lag_col] = recent_temps_for_lags[12 - lag_num]
                        else:
                            features_future[lag_col] = np.nan
                            missing_a_lag = True
                    
                    if missing_a_lag:
                        print(f"Could not form complete feature vector for {month_name} 2025 prediction due to missing lag data in iterative step.")
                        break

                    features_future_df = pd.DataFrame([features_future])
                    features_future_df_sm = sm.add_constant(features_future_df[best_combo_mlr], has_constant='add')
                    features_future_df_sm = features_future_df_sm[cols_for_pred] # Ensure order
                    
                    predicted_temp_future = best_model_mlr_results.predict(features_future_df_sm)[0]
                    diff_future = predicted_temp_future - actual_future_temp
                    print(f"  {month_name} 2025: Predicted={predicted_temp_future:.2f}°C, Actual={actual_future_temp:.1f}°C, Diff={diff_future:.2f}°C")
                    
                    temps_for_future_lags.append(actual_future_temp)
                print("Comment Q17 (Future): The model's performance for Feb-Apr 2025 depends on its ability to generalize. Using actuals for lags tests one-step ahead forecast capability.")
        else: # Not enough data in full_temp_data for last_12_temps
            print("Not enough historical data in full_temp_data to make Jan 2025 MLR prediction.")
            q_answers['q17'] = "N/A (Insufficient historical data)"
            
    elif not best_model_mlr_results or not best_combo_mlr :
        print("No optimal multivariate model from Q16 (or no best_combo_mlr) to make predictions for Q17.")
        q_answers['q17'] = "N/A (No Q16 model)"
    elif full_temp_data.empty:
        print("full_temp_data is empty, cannot make predictions for Q17.")
        q_answers['q17'] = "N/A (full_temp_data empty)"


# --- Generate content for template.csv ---
print("\n\n--- Answers for template.csv ---")
# Ensure all q_answers are strings for CSV
template_csv_content = "Question_ID,Answer\n"
processed_q_main_numbers = set()

for q_id_num in range(1, 18): 
    # Handle questions with specific sub-parts (a, b, c, etc.)
    sub_parts_map = {
        1: ['a', 'b'],
        2: [chr(ord('a') + i) for i in range(8)], # a-h
        3: ['a', 'b', 'c', 'd'],
        4: ['a', 'b', 'c'],
        5: ['a', 'b', 'c'],
        6: ['a', 'b', 'c'],
        8: ['a', 'b'],
        12: ['a', 'b', 'c', 'd', 'e'],
        13: ['a', 'b'],
        14: ['a', 'b'],
        16: ['a', 'b'] # q16a, q16b are primary. Others like q16_adj_r2 are for report.
    }

    if q_id_num in sub_parts_map:
        if q_id_num not in processed_q_main_numbers: # Ensure main number isn't processed twice
            for part in sub_parts_map[q_id_num]:
                q_key = f"q{q_id_num}{part}"
                answer = q_answers.get(q_key, 'Not Computed')
                # Format floats to consistent precision if they are indeed floats
                if isinstance(answer, float):
                    answer = f"{answer:.4f}" # Example: 4 decimal places
                template_csv_content += f"{q_key},{str(answer)}\n"
            processed_q_main_numbers.add(q_id_num)
    else: # Single part questions (q7, q9, q10, q11, q15, q17)
        if q_id_num not in processed_q_main_numbers:
            q_key = f"q{q_id_num}" # e.g. q7
            answer = q_answers.get(q_key, 'Not Computed')
            if isinstance(answer, float):
                 answer = f"{answer:.4f}"
            template_csv_content += f"{q_key},{str(answer)}\n"
            processed_q_main_numbers.add(q_id_num)


print(template_csv_content)
try:
    with open("template_answers.csv", "w") as f:
        f.write(template_csv_content)
    print("\nAnswers saved to template_answers.csv")
except Exception as e_write_csv:
    print(f"\nError writing template_answers.csv: {e_write_csv}")


print("\nPython script execution finished.")
print("Remember to check the saved plot images (.png files) for your report.")
print("Ensure all [qij] values in your LaTeX report are filled from the console output or template_answers.csv.")

