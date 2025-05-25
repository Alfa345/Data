################################################################################
# 0. SETUP AND CONFIGURATION
################################################################################

# --- 0.1 Import Libraries ---
# Pandas for data manipulation and analysis (e.g., DataFrames)
import pandas as pd
# NumPy for numerical operations, especially with arrays
import numpy as np
# Matplotlib for plotting
import matplotlib.pyplot as plt
# Seaborn for enhanced statistical visualizations
import seaborn as sns
# StandardScaler for feature scaling (centering and reducing data)
from sklearn.preprocessing import StandardScaler
# PCA for Principal Component Analysis
from sklearn.decomposition import PCA
# Statsmodels for statistical models, including OLS (Ordinary Least Squares) for regression
import statsmodels.api as sm
# itertools.combinations for generating combinations of variables in multivariate regression
from itertools import combinations
# OS module for interacting with the operating system (e.g., paths, directory creation)
import os

# --- 0.2 Global Variables and Configuration ---
# Dictionary to store answers for the final CSV report
q_answers = {}

# Matplotlib and Seaborn visual theme configuration
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6) # Default figure size
plt.rcParams['font.size'] = 12         # Default font size

# --- 0.3 Directory Configuration ---
# Determine the base directory (where this script is located)
# This makes file paths relative to the script, improving portability.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SETS_DIR = os.path.join(BASE_DIR, "data sets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# --- 0.4 Helper Functions ---
def create_results_directory_if_not_exists(directory_path):
    """
    Creates a directory if it doesn't already exist.
    This is useful for organizing output files like plots and CSVs.

    Args:
        directory_path (str): The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"INFO: Created directory: {directory_path}")

def save_plot(filename, tight_layout=True):
    """
    Saves the current matplotlib plot to the pre-configured RESULTS_DIR.
    It also closes the plot to free up memory, which is good practice when generating many plots.

    Args:
        filename (str): The name of the file to save the plot as (e.g., "my_plot.png").
        tight_layout (bool): Whether to apply plt.tight_layout() before saving.
    """
    output_filepath = os.path.join(RESULTS_DIR, filename)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(output_filepath)
    print(f"PLOT: Plot saved as {output_filepath}")
    plt.close()


################################################################################
# 1. DATA LOADING AND PREPROCESSING
################################################################################

def load_and_preprocess_data1(filepath, city_col_name, rename_map, expected_processing_cols):
    """
    Loads and preprocesses data1.csv.
    - Handles different delimiters (comma, semicolon).
    - Renames columns to a standard format.
    - Sets the specified city column as the DataFrame index.
    - Converts specified weather variable columns to numeric types.

    Args:
        filepath (str): The full path to the data1.csv file.
        city_col_name (str): The name of the column containing city names to be used as index.
        rename_map (dict): A dictionary to map original column names to desired names.
        expected_processing_cols (list): A list of column names (after renaming)
                                         that should be converted to numeric.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame, or None if loading fails.
    """
    print(f"\n--- Loading and Preprocessing {os.path.basename(filepath)} ---")
    try:
        # Try loading with comma delimiter
        df = pd.read_csv(filepath)
        print(f"INFO: Successfully loaded {filepath} with default comma delimiter.")
    except FileNotFoundError:
        print(f"ERROR: {filepath} not found. Please check the path.")
        return None
    except Exception:
        print(f"INFO: Failed to load {filepath} with comma. Trying semicolon delimiter...")
        try:
            # Fallback to semicolon delimiter
            df = pd.read_csv(filepath, sep=';')
            print(f"INFO: Successfully loaded {filepath} with semicolon delimiter.")
        except Exception as e_semi:
            print(f"ERROR: Failed to load {filepath} with semicolon delimiter: {e_semi}")
            return None

    # Standardize column names: remove leading/trailing spaces and apply rename_map
    df.columns = df.columns.str.strip()
    df.rename(columns=rename_map, inplace=True)
    print(f"INFO: Columns after renaming: {df.columns.tolist()}")

    # Set the 'City' column as the index
    if city_col_name in df.columns:
        df.set_index(city_col_name, inplace=True)
        print(f"INFO: '{city_col_name}' set as index.")
    else:
        print(f"ERROR: City column '{city_col_name}' not found after renaming. Cannot set index.")
        print(f"       Available columns: {df.columns.tolist()}")
        return None # Critical error, cannot proceed with this DataFrame

    # Convert specified weather variable columns to numeric
    # Errors during conversion will be set to NaN (Not a Number)
    print("INFO: Converting weather columns to numeric...")
    for col in expected_processing_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # This warning is important if a key analysis column is missing.
            print(f"WARNING: Expected processing column '{col}' not found for numeric conversion.")
    print("INFO: Numeric conversion attempt complete.")
    return df

def load_and_preprocess_data2(filepath, rename_map, required_cols, month_order_english):
    """
    Loads and preprocesses data2.csv.
    - Handles different delimiters.
    - Renames columns.
    - Converts temperature to numeric.
    - Creates a 'month_ID' (0-11) from the 'Month' column.
    - Converts 'Year' to numeric.

    Args:
        filepath (str): The full path to the data2.csv file.
        rename_map (dict): Dictionary to map original column names to desired names.
        required_cols (list): List of column names essential for processing (after renaming).
        month_order_english (list): List of English month names for mapping.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame, or None if loading or critical preprocessing fails.
    """
    print(f"\n--- Loading and Preprocessing {os.path.basename(filepath)} ---")
    try:
        df = pd.read_csv(filepath)
        print(f"INFO: Successfully loaded {filepath} with default comma delimiter.")
    except FileNotFoundError:
        print(f"ERROR: {filepath} not found. Please check the path.")
        return None
    except Exception:
        print(f"INFO: Failed to load {filepath} with comma. Trying semicolon delimiter...")
        try:
            df = pd.read_csv(filepath, sep=';')
            print(f"INFO: Successfully loaded {filepath} with semicolon delimiter.")
        except Exception as e_semi:
            print(f"ERROR: Failed to load {filepath} with semicolon delimiter: {e_semi}")
            return None

    df.columns = df.columns.str.strip()
    if rename_map: # Only rename if a map is provided
        df.rename(columns=rename_map, inplace=True)
        print(f"INFO: Columns after renaming: {df.columns.tolist()}")
    else:
        print(f"INFO: No rename map for {os.path.basename(filepath)} specified. Using original column names (stripped).")


    # Verify required columns exist
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Not all required columns {required_cols} found in {os.path.basename(filepath)}.")
        print(f"       Actual columns: {df.columns.tolist()}")
        print(f"       Please check 'rename_map_data2' or your {os.path.basename(filepath)} file.")
        return None # Critical error

    # Convert 'Max_Temperature_Paris' to numeric
    temp_col = 'Max_Temperature_Paris' # Assuming this is in required_cols after renaming
    if temp_col in df.columns:
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        print(f"INFO: '{temp_col}' column converted to numeric.")
    else:
        # This case should be caught by the required_cols check above, but defensive check here.
        print(f"WARNING: '{temp_col}' column not found for numeric conversion.")

    # Create 'month_ID' (0 for Jan to 11 for Dec)
    df['month_ID'] = np.nan # Initialize column
    if 'Month' in df.columns:
        print(f"INFO: Processing 'Month' column (dtype: {df['Month'].dtype}).")
        if pd.api.types.is_numeric_dtype(df['Month']): # If month is already 1-12
            df['month_ID'] = df['Month'] - 1
            print("INFO: Numeric 'Month' column converted to 0-11 'month_ID'.")
        else: # If month is string names
            # Example: For French months, you'd map them to English first if needed
            # month_name_map_to_english = {'janvier': 'January', ...}
            # df['Month_English'] = df['Month'].str.lower().str.strip().map(month_name_map_to_english)
            # current_month_series = df['Month_English']
            current_month_series = df['Month'].astype(str).str.lower().str.strip()
            month_to_id_map = {name.lower(): i for i, name in enumerate(month_order_english)}
            df['month_ID'] = current_month_series.map(month_to_id_map)
            if df['month_ID'].isnull().any():
                print("WARNING: Some 'Month' values could not be mapped to 'month_ID'. Check for non-standard month names.")
            else:
                print("INFO: String 'Month' column successfully mapped to 'month_ID'.")
    else:
        print("WARNING: 'Month' column not found. 'month_ID' cannot be created.")

    # Convert 'Year' to numeric
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        print("INFO: 'Year' column converted to numeric.")

    return df


################################################################################
# 2. PRELIMINARY ANALYSIS (data1.csv)
################################################################################
# This section focuses on understanding the basic characteristics of the weather data from data1.csv

def perform_q1_initial_data_stats(df, q_answers_dict):
    """
    Answers Q1: Calculates initial number of cities and cities with missing data.
    Modifies the DataFrame by dropping rows with any missing data.

    Args:
        df (pandas.DataFrame): The input DataFrame (expected to be data1).
        q_answers_dict (dict): Dictionary to store answers.

    Returns:
        pandas.DataFrame: The DataFrame after removing rows with missing data.
    """
    print("\n--- Q1: Initial Data and Missing Values ---")
    initial_cities = df.shape[0]
    q_answers_dict['q1a'] = initial_cities
    print(f"[q1a] Initial number of cities (rows): {initial_cities}")

    cities_with_missing = df[df.isnull().any(axis=1)]
    num_cities_with_missing = cities_with_missing.shape[0]
    q_answers_dict['q1b'] = num_cities_with_missing
    print(f"[q1b] Number of cities affected by missing measurements: {num_cities_with_missing}")

    if num_cities_with_missing > 0:
        print(f"INFO: Cities with missing data (before removal): {cities_with_missing.index.tolist()}")
        df.dropna(inplace=True) # Modifies df in place
        print(f"INFO: Cities with missing data removed. Remaining cities: {df.shape[0]}")
    else:
        print("INFO: No cities with missing data found.")
    return df

def perform_q2_extreme_values(df, weather_vars_map, q_labels, q_answers_dict):
    """
    Answers Q2: Finds cities associated with minimum and maximum values for specified weather variables.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        weather_vars_map (dict): Maps variable names to descriptions (low, high).
        q_labels (list): List of question labels (e.g., 'q2a', 'q2b', ...).
        q_answers_dict (dict): Dictionary to store answers.
    """
    print("\n--- Q2: Extreme Weather Values by City ---")
    # This analysis helps identify cities that represent climatic extremes in the dataset.
    label_idx = 0
    for var, (desc_low, desc_high) in weather_vars_map.items():
        q_key_low = q_labels[label_idx]
        q_key_high = q_labels[label_idx+1]

        if var in df.columns and not df[var].empty and df[var].notna().any():
            min_val = df[var].min()
            min_city = df[var].idxmin() # idxmin() returns the index (City name)
            q_answers_dict[q_key_low] = f"{min_city} ({min_val})"
            print(f"[{q_key_low}] City with {desc_low}: {min_city} ({min_val})")

            max_val = df[var].max()
            max_city = df[var].idxmax() # idxmax() returns the index (City name)
            q_answers_dict[q_key_high] = f"{max_city} ({max_val})"
            print(f"[{q_key_high}] City with {desc_high}: {max_city} ({max_val})")
        else:
            print(f"WARNING: Column '{var}' not found, empty, or all NaN for Q2 analysis. Skipping.")
            q_answers_dict[q_key_low] = "N/A (Column problem)"
            q_answers_dict[q_key_high] = "N/A (Column problem)"
        label_idx += 2
    print("COMMENT Q2: These values highlight regional climatic differences across the sampled cities.")

def perform_q3_variances(df, expected_cols, q_answers_dict):
    """
    Answers Q3: Computes the variance for specified weather variables.
    Variance measures the spread or dispersion of data points around their mean.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        expected_cols (list): List of weather variable columns to calculate variance for.
        q_answers_dict (dict): Dictionary to store answers.

    Returns:
        pandas.Series: A Series containing the variances, or an empty Series if not calculable.
    """
    print("\n--- Q3: Variance of Weather Variables ---")
    # Variance gives an idea of how much the weather conditions differ from city to city for each variable.
    # We only consider numeric columns that are part of our expected weather variables.
    numeric_cols_for_var = df.select_dtypes(include=np.number).columns.intersection(expected_cols)
    variances = pd.Series(dtype=float) # Initialize as empty

    if not numeric_cols_for_var.empty and df[numeric_cols_for_var].notna().any().any():
        variances = df[numeric_cols_for_var].var()
        print("INFO: Calculated variances:")
        var_map_q3 = {
            'Temp_min_C': 'q3a', 'Temp_max_C': 'q3b',
            'Precipitation_mm': 'q3c', 'Ensoleillement_h': 'q3d'
        }
        for var_name, q_id in var_map_q3.items():
            if var_name in variances and pd.notna(variances[var_name]):
                q_answers_dict[q_id] = variances[var_name]
                print(f"[{q_id}] Variance of {var_name}: {q_answers_dict[q_id]:.2f}")
            else:
                q_answers_dict[q_id] = "N/A"
                print(f"[{q_id}] Variance of {var_name}: N/A (Not calculable or missing)")
    else:
        print("WARNING: No numeric columns with non-NaN values found to calculate variances for Q3.")
        for q_id_suffix in ['a', 'b', 'c', 'd']: q_answers_dict[f'q3{q_id_suffix}'] = "N/A"

    print("COMMENT Q3: Variables with higher variance (e.g., Rainfall, Sunshine) show greater differences across cities than those with lower variance (e.g., Temperatures).")
    return variances

def perform_q4_q5_min_max_variance_stats_and_histograms(df, variances, q_answers_dict):
    """
    Answers Q4 & Q5: Computes descriptive statistics (mean, median, std) and plots histograms
    for the variables with the lowest and highest variance.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        variances (pandas.Series): Series of variances calculated in Q3.
        q_answers_dict (dict): Dictionary to store answers.
    """
    if variances.empty or not variances.notna().any():
        print("WARNING: Skipping Q4 & Q5 as variance data is empty or all NaN.")
        for q_id in ['q4a', 'q4b', 'q4c', 'q5a', 'q5b', 'q5c']: q_answers_dict[q_id] = "N/A"
        return

    # Identify variables with lowest and highest variance
    var_lowest_variance_name = variances.idxmin()
    var_highest_variance_name = variances.idxmax()

    # --- Q4: Stats for Variable with Lowest Variance ---
    print(f"\n--- Q4: Stats for '{var_lowest_variance_name}' (Lowest Variance) ---")
    # This variable shows the most consistent measurements across different cities.
    data_low_var = df[var_lowest_variance_name]
    q_answers_dict['q4a'] = data_low_var.mean()
    q_answers_dict['q4b'] = data_low_var.median()
    q_answers_dict['q4c'] = data_low_var.std()
    print(f"[q4a] Mean: {q_answers_dict['q4a']:.2f}")
    print(f"[q4b] Median: {q_answers_dict['q4b']:.2f}")
    print(f"[q4c] Standard Deviation: {q_answers_dict['q4c']:.2f}")

    plt.figure() # Ensure a new figure for the histogram
    data_low_var.hist(bins=15, edgecolor='black')
    plt.title(f"Histogram of {var_lowest_variance_name}")
    plt.xlabel(var_lowest_variance_name)
    plt.ylabel("Frequency")
    save_plot(f"histogram_{var_lowest_variance_name.replace('/', '_').replace(' ', '_')}.png")
    print(f"COMMENT Q4: The histogram for {var_lowest_variance_name} shows its distribution. If mean and median are close, the distribution is likely symmetric.")

    # --- Q5: Stats for Variable with Highest Variance ---
    print(f"\n--- Q5: Stats for '{var_highest_variance_name}' (Highest Variance) ---")
    # This variable shows the most diverse measurements across cities.
    data_high_var = df[var_highest_variance_name]
    q_answers_dict['q5a'] = data_high_var.mean()
    q_answers_dict['q5b'] = data_high_var.median()
    q_answers_dict['q5c'] = data_high_var.std()
    print(f"[q5a] Mean: {q_answers_dict['q5a']:.2f}")
    print(f"[q5b] Median: {q_answers_dict['q5b']:.2f}")
    print(f"[q5c] Standard Deviation: {q_answers_dict['q5c']:.2f}")

    plt.figure() # Ensure a new figure
    data_high_var.hist(bins=15, edgecolor='black')
    plt.title(f"Histogram of {var_highest_variance_name}")
    plt.xlabel(var_highest_variance_name)
    plt.ylabel("Frequency")
    save_plot(f"histogram_{var_highest_variance_name.replace('/', '_').replace(' ', '_')}.png")
    print(f"COMMENT Q5: The histogram for {var_highest_variance_name} is expected to be more spread out, reflecting its higher variance.")

def perform_q6_variable_correlations(df, expected_cols, q_answers_dict):
    """
    Answers Q6: Calculates and visualizes linear correlations between different weather variables.
    Correlation indicates the strength and direction of a linear relationship between two variables.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        expected_cols (list): List of weather variable columns to correlate.
        q_answers_dict (dict): Dictionary to store answers.
    """
    print("\n--- Q6: Linear Correlations Between Weather Variables ---")
    # We want to see how weather variables relate to each other (e.g., does higher temperature correlate with more sunshine?)
    numeric_cols_for_corr = df.select_dtypes(include=np.number).columns.intersection(expected_cols)

    if len(numeric_cols_for_corr) < 2 or not df[numeric_cols_for_corr].notna().any().any():
        print("WARNING: Skipping Q6 as not enough numeric weather variables or valid data for correlation.")
        for q_id_suffix in ['a', 'b', 'c']: q_answers_dict[f'q6{q_id_suffix}'] = "N/A"
        return

    correlation_matrix_vars = df[numeric_cols_for_corr].corr()
    # To find pairs easily, unstack the matrix, sort, and remove self-correlations and duplicates
    corr_pairs = correlation_matrix_vars.unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)] # Remove self (VarA vs VarA)
    corr_pairs = corr_pairs.iloc[::2] # Remove duplicates (VarA-VarB is same as VarB-VarA after sorting)

    if corr_pairs.empty or not corr_pairs.notna().any():
        print("WARNING: No valid correlation pairs found for Q6.")
        for q_id_suffix in ['a', 'b', 'c']: q_answers_dict[f'q6{q_id_suffix}'] = "N/A"
        return

    # Most positively correlated
    most_pos_pair = corr_pairs.index[0]
    most_pos_val = corr_pairs.iloc[0]
    q_answers_dict['q6a'] = f"{most_pos_pair[0]} & {most_pos_pair[1]} ({most_pos_val:.2f})"
    print(f"[q6a] Two most positively correlated variables: {most_pos_pair[0]} and {most_pos_pair[1]}, Correlation: {most_pos_val:.2f}")

    # Most negatively correlated
    most_neg_pair = corr_pairs.index[-1]
    most_neg_val = corr_pairs.iloc[-1]
    q_answers_dict['q6b'] = f"{most_neg_pair[0]} & {most_neg_pair[1]} ({most_neg_val:.2f})"
    print(f"[q6b] Two most negatively correlated variables: {most_neg_pair[0]} and {most_neg_pair[1]}, Correlation: {most_neg_val:.2f}")

    # Least correlated (closest to zero)
    least_corr_abs_series = corr_pairs.abs().sort_values()
    least_corr_abs_pair = least_corr_abs_series.index[0]
    # Get original signed value from the full correlation matrix
    original_least_corr_val = correlation_matrix_vars.loc[least_corr_abs_pair[0], least_corr_abs_pair[1]]
    q_answers_dict['q6c'] = f"{least_corr_abs_pair[0]} & {least_corr_abs_pair[1]} ({original_least_corr_val:.2f})"
    print(f"[q6c] Two least correlated variables (closest to 0): {least_corr_abs_pair[0]} and {least_corr_abs_pair[1]}, Correlation: {original_least_corr_val:.2f}")

    # Scatter plots to visualize these relationships
    plots_to_make = [
        (most_pos_pair[0], most_pos_pair[1], "Most_Positive_Correlation"),
        (most_neg_pair[0], most_neg_pair[1], "Most_Negative_Correlation"),
        (least_corr_abs_pair[0], least_corr_abs_pair[1], "Least_Correlation")
    ]
    for var1, var2, title_suffix in plots_to_make:
        if var1 in df.columns and var2 in df.columns:
            plt.figure(figsize=(10, 7)) # Adjusted size for potentially many city names
            sns.scatterplot(x=df[var1], y=df[var2])
            # Annotate each point with the city name (index of the DataFrame)
            for city_name, row_data in df.iterrows():
                plt.text(row_data[var1], row_data[var2], city_name, fontsize=8, alpha=0.7)
            plt.title(f"Scatter: {var1} vs {var2} ({title_suffix})")
            plt.xlabel(var1); plt.ylabel(var2)
            save_plot(f"scatter_{var1.replace('/', '_')}_{var2.replace('/', '_')}.png")
    print("COMMENT Q6: Positive correlation means variables tend to increase together. Negative means one tends to decrease as the other increases. Scatter plots help visualize these trends and identify outliers.")

def perform_q7_city_correlations(df, expected_cols, q_answers_dict):
    """
    Answers Q7: Calculates and visualizes linear correlations between cities based on their weather profiles.
    A heatmap is used for visualization.

    Args:
        df (pandas.DataFrame): The input DataFrame (cities as index, weather variables as columns).
        expected_cols (list): List of weather variable columns to use for comparing cities.
        q_answers_dict (dict): Dictionary to store answers.
    """
    print("\n--- Q7: Linear Correlations Between Cities ---")
    # Here, we want to see which cities have similar overall weather patterns.
    # We correlate cities based on their values for the weather variables.
    # Need to select only numeric weather columns for this.
    numeric_cols_for_city_corr = df.select_dtypes(include=np.number).columns.intersection(expected_cols)

    if len(numeric_cols_for_city_corr) == 0 or df.shape[0] < 2 or not df[numeric_cols_for_city_corr].notna().any().any():
        print("WARNING: Skipping Q7 as not enough numeric data or cities for correlation.")
        q_answers_dict['q7'] = "N/A (Insufficient data)"
        return

    # To correlate cities, we transpose the DataFrame so cities become columns,
    # then .corr() calculates column-wise correlations.
    correlation_matrix_cities = df[numeric_cols_for_city_corr].T.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix_cities, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix Between Cities (based on weather variables)")
    save_plot("correlation_matrix_cities.png")
    q_answers_dict['q7'] = "Heatmap 'correlation_matrix_cities.png' generated."
    print("COMMENT Q7: The heatmap visualizes similarities. Cities with similar colors (e.g., dark red for high positive correlation) have similar weather profiles according to the selected variables.")


# ... (Further parts will follow the same refactoring pattern) ...

# This is a start to show the refactoring approach for the initial parts.
# The complete script would refactor Parts 2, 3, 4, and the final CSV generation similarly.
# Due to length constraints, I'll provide the rest of the refactored parts conceptually
# and then the main orchestration block.

################################################################################
# 3. PRINCIPAL COMPONENT ANALYSIS (PCA) (data1.csv)
################################################################################
# PCA is a dimensionality reduction technique. It transforms the original correlated
# weather variables into a smaller set of uncorrelated variables called principal components (PCs),
# while retaining most of the original information (variance).

def perform_pca_analysis_q8_q10(df_original, expected_cols, q_answers_dict):
    """
    Performs PCA (Q8, Q9, Q10) on the provided data.
    - Selects numeric features and scales them.
    - Applies PCA to get the first two principal components.
    - Generates PCA scatter plot of cities (Q8).
    - Generates correlation circle (Q9).
    - Generates biplot (Q10).

    Args:
        df_original (pandas.DataFrame): Original DataFrame (data1) with cities as index.
        expected_cols (list): List of weather variable columns to use for PCA.
        q_answers_dict (dict): Dictionary to store answers.
    """
    print("\n--- Part 2 (PCA): Principal Component Analysis ---")
    df_numeric = df_original.select_dtypes(include=np.number)
    df_numeric = df_numeric[df_numeric.columns.intersection(expected_cols)]

    if df_numeric.shape[1] < 2 or not df_numeric.notna().any().any():
        print("WARNING: Not enough numeric features or valid data for PCA. Skipping Part 2.")
        for q_id in ['q8a', 'q8b', 'q9', 'q10']: q_answers_dict[q_id] = "N/A"
        return

    print(f"INFO: Performing PCA on {df_numeric.shape[1]} features: {df_numeric.columns.tolist()}")
    # Step 1: Center and reduce data (Standardization)
    # This is crucial for PCA because it's sensitive to variable scales.
    # StandardScaler transforms data to have zero mean and unit variance.
    scaler = StandardScaler()
    data_scaled_array = scaler.fit_transform(df_numeric)
    # For clarity, convert scaled array back to DataFrame with original index and columns
    data_scaled_df = pd.DataFrame(data_scaled_array, columns=df_numeric.columns, index=df_numeric.index)

    # Step 2: Apply PCA
    # We are interested in the first two principal components as per the project.
    pca = PCA(n_components=2)
    principal_components_array = pca.fit_transform(data_scaled_df)
    pc_df = pd.DataFrame(data=principal_components_array, columns=['PC1', 'PC2'], index=data_scaled_df.index)

    # --- Q8: PCA Scatter Plot and Explained Variance ---
    print("\n--- Q8: PCA on Weather Data (Scores Plot) ---")
    # The PCs capture the directions of maximum variance in the data.
    # PC1 explains the most variance, PC2 the second most (orthogonal to PC1).
    explained_variance_ratio = pca.explained_variance_ratio_
    q_answers_dict['q8a'] = explained_variance_ratio[0] * 100
    q_answers_dict['q8b'] = explained_variance_ratio[1] * 100
    print(f"[q8a] Percentage of variance explained by PC1: {q_answers_dict['q8a']:.2f}%")
    print(f"[q8b] Percentage of variance explained by PC2: {q_answers_dict['q8b']:.2f}%")

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', data=pc_df, s=60, hue=pc_df.index, legend=False) # Color by city for distinction
    for city_name, row in pc_df.iterrows():
        plt.text(row['PC1'] + 0.05, row['PC2'] + 0.05, city_name, fontsize=9)
    plt.xlabel(f"Principal Component 1 ({q_answers_dict['q8a']:.2f}%)")
    plt.ylabel(f"Principal Component 2 ({q_answers_dict['q8b']:.2f}%)")
    plt.title("PCA: Cities Projected onto First Two Principal Components")
    plt.axhline(0, color='grey', lw=0.5, linestyle='--'); plt.axvline(0, color='grey', lw=0.5, linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.7)
    save_plot("pca_cities.png")
    print("COMMENT Q8: This plot (scores plot) shows how cities are positioned relative to each other based on the two main dimensions of variation in their weather data. Cities close together have similar weather profiles as captured by PC1 and PC2.")

    # --- Q9: Correlation Circle (Loadings Plot) ---
    print("\n--- Q9: PCA Correlation Circle ---")
    # The correlation circle shows how the original variables contribute to the principal components.
    # Loadings are the correlations between original variables and PCs.
    # pca.components_ has shape (n_components, n_features). We need (n_features, n_components).
    loadings = pca.components_.T

    fig, ax = plt.subplots(figsize=(8, 8))
    for i, var_name in enumerate(df_numeric.columns): # Use columns from the data fed to PCA
        # Arrows represent the original variables in the PC space.
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                 head_width=0.05, head_length=0.05, fc='r', ec='r', length_includes_head=True)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var_name,
                color='r', ha='center', va='center', fontsize=10)
    # The unit circle helps interpret: variables near the circle are well-represented by these 2 PCs.
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    ax.add_artist(circle)
    ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel(f"PC1 ({q_answers_dict['q8a']:.2f}%)"); ax.set_ylabel(f"PC2 ({q_answers_dict['q8b']:.2f}%)")
    ax.set_title("PCA Correlation Circle (Variable Loadings)")
    ax.axhline(0, color='grey', lw=0.5, linestyle='--'); ax.axvline(0, color='grey', lw=0.5, linestyle='--')
    ax.grid(True, linestyle=':', alpha=0.7); ax.set_aspect('equal', adjustable='box')
    save_plot("correlation_circle.png")
    q_answers_dict['q9'] = "Plot 'correlation_circle.png' generated."
    print("COMMENT Q9: Variables with arrows pointing in similar directions are positively correlated. Opposite directions mean negative correlation. Length of arrow indicates how much that variable contributes to the PCs shown.")

    # --- Q10: Biplot (Superimposing Scores and Loadings) ---
    print("\n--- Q10: PCA Biplot (City Scores and Variable Loadings) ---")
    # A biplot combines the scores (cities) and loadings (variables) on the same graph.
    # This helps interpret which variables are driving the separation of cities in the PC space.
    fig, ax1 = plt.subplots(figsize=(14, 10))
    # Plot city scores (from Q8)
    ax1.scatter(pc_df['PC1'], pc_df['PC2'], c='blue', label='Cities', alpha=0.7, s=50)
    for city_name, row in pc_df.iterrows():
        ax1.text(row['PC1'] + 0.05, row['PC2'] + 0.05, city_name, fontsize=9, c='blue')
    ax1.set_xlabel(f"Principal Component 1 ({q_answers_dict['q8a']:.2f}%) - City Scores")
    ax1.set_ylabel(f"Principal Component 2 ({q_answers_dict['q8b']:.2f}%) - City Scores", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axhline(0, color='grey', lw=0.5, linestyle='--'); ax1.axvline(0, color='grey', lw=0.5, linestyle='--')
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Create a second y-axis for loadings (from Q9)
    ax2 = ax1.twinx().twiny() # Share x and y axes for overlay
    for i, var_name in enumerate(df_numeric.columns):
        ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                  head_width=0.03, head_length=0.03, fc='red', ec='red', alpha=0.8, length_includes_head=True)
        ax2.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, var_name,
                 color='red', ha='center', va='center', fontsize=10, alpha=0.8)
    ax2.set_xlim([-1.2, 1.2]); ax2.set_ylim([-1.2, 1.2]) # Scale for loadings
    ax2.set_xticks([]); ax2.set_yticks([]) # Hide ticks for the second axes to avoid clutter

    plt.title("PCA Biplot: City Scores (Blue) and Variable Loadings (Red)")
    save_plot("pca_biplot.png")
    q_answers_dict['q10'] = "Plot 'pca_biplot.png' generated."
    print("COMMENT Q10: Cities are influenced by variables whose arrows point towards them. For Q2, cities at the extremes of the plot, aligned with specific variable arrows, might correspond to min/max values for those variables (e.g., a city far along an axis strongly correlated with 'Temp_max_C' could be the one with highest max temp).")


################################################################################
# 4. SIMPLE LINEAR REGRESSION (SLR) (data2.csv - Paris 2024)
################################################################################
# SLR aims to model the linear relationship between one independent variable (month_ID)
# and one dependent variable (Max_Temperature_Paris). Model: Temp = B0 + B1*month_ID + error.

def perform_slr_analysis_q11_q14(data2_full, target_year, month_order_english_list, q_answers_dict):
    """
    Performs Simple Linear Regression analysis (Q11-Q14) for a target year.
    - Filters data for the target year and valid months/temperatures.
    - Q11: Plots temperature evolution.
    - Q12: Finds optimal 'n' (number of past months) for SLR based on adjusted R-squared,
           fits the model, and plots it.
    - Q13: Predicts temperature for Jan of the next year.
    - Q14: Performs hypothesis test on the slope coefficient (Beta1).

    Args:
        data2_full (pandas.DataFrame): The preprocessed data2 DataFrame.
        target_year (int): The year to focus on for SLR (e.g., 2024).
        month_order_english_list (list): List of English month names for plotting.
        q_answers_dict (dict): Dictionary to store answers.

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The best SLR model, or None.
    """
    print(f"\n--- Part 3 (SLR): Simple Linear Regression for Paris {target_year} ---")
    # Filter data for the target year and ensure necessary columns are valid
    df_year = data2_full[
        (data2_full['Year'] == target_year) &
        pd.notna(data2_full['month_ID']) &
        pd.notna(data2_full['Max_Temperature_Paris'])
    ].copy() # Use .copy() to avoid SettingWithCopyWarning
    df_year.sort_values('month_ID', inplace=True)
    df_year.reset_index(drop=True, inplace=True)
    print(f"INFO: Filtered data for Paris {target_year}: {len(df_year)} valid monthly records found.")

    # --- Q11: Temperature Evolution Plot ---
    print(f"\n--- Q11: Temperature Evolution in Paris ({target_year}) ---")
    if not df_year.empty and df_year['month_ID'].notna().any():
        plt.figure(figsize=(10, 6))
        plt.plot(df_year['month_ID'], df_year['Max_Temperature_Paris'], marker='o', linestyle='-')
        plt.xticks(ticks=range(12), labels=month_order_english_list, rotation=45, ha="right")
        plt.xlabel("Month"); plt.ylabel(f"Max Temperature (°C) in Paris ({target_year})")
        plt.title(f"Evolution of Max Temperature in Paris ({target_year})")
        save_plot(f"paris_temp_{target_year}.png")
        q_answers_dict['q11'] = f"Plot 'paris_temp_{target_year}.png' generated."
        print(f"COMMENT Q11: The plot shows the typical seasonal temperature pattern for Paris in {target_year}.")
    else:
        print(f"WARNING: Skipping Q11 plot for {target_year} due to insufficient valid data.")
        q_answers_dict['q11'] = "N/A (Insufficient data)"

    # --- Q12-Q14: Optimal SLR Model, Prediction, Hypothesis Test ---
    # These steps require at least 2 data points for regression.
    best_slr_model_results = None
    if len(df_year) < 2:
        print(f"WARNING: Not enough data for Paris {target_year} to perform SLR. Skipping Q12-Q14.")
        for q_id_sfx in ['a','b','c','d','e']: q_answers_dict[f'q12{q_id_sfx}'] = "N/A"
        for q_id_sfx in ['a','b']: q_answers_dict[f'q13{q_id_sfx}'] = "N/A"; q_answers_dict[f'q14{q_id_sfx}'] = "N/A"
        return None

    print(f"\n--- Q12: Optimal Simple Linear Regression for Paris {target_year} Temps ---")
    # We want to find the best 'n' (number of most recent months from df_year)
    # that gives the highest adjusted R-squared for the model: Temp = B0 + B1*month_ID.
    best_n_slr = -1
    best_adj_r2_slr = -float('inf')
    # Store parameters of the best model
    optimal_b0_slr, optimal_b1_slr, optimal_r2_slr = None, None, None

    # 'n' ranges from 1 to number of available months in df_year (max 12).
    # Regression needs at least 2 points. df_model for SLR is 1 (slope). const is 1. total 2 params.
    # So, n_months_slr (number of observations) must be >= 2.
    for n_months_slr in range(2, len(df_year) + 1):
        current_data_slr = df_year.tail(n_months_slr) # Use the last 'n' months
        X_slr = current_data_slr['month_ID']
        y_slr = current_data_slr['Max_Temperature_Paris']
        # Add a constant for the intercept term (B0)
        X_slr_sm = sm.add_constant(X_slr)

        try:
            model_slr = sm.OLS(y_slr, X_slr_sm).fit()
            # Adjusted R-squared penalizes for adding predictors that don't improve the model.
            # It's better for comparing models with different numbers of predictors (though here only 'n' changes the data window).
            if model_slr.rsquared_adj > best_adj_r2_slr:
                best_adj_r2_slr = model_slr.rsquared_adj
                best_n_slr = n_months_slr
                best_slr_model_results = model_slr # Store the fitted model object
                optimal_b0_slr = model_slr.params['const']
                optimal_b1_slr = model_slr.params['month_ID'] # 'month_ID' is the name of our predictor
                optimal_r2_slr = model_slr.rsquared
        except Exception as e_ols:
            # print(f"DEBUG: Error during OLS for n_months_slr={n_months_slr}: {e_ols}")
            continue # If a specific 'n' fails, try the next one

    if best_slr_model_results:
        q_answers_dict['q12a'] = best_n_slr
        q_answers_dict['q12b'] = best_adj_r2_slr
        q_answers_dict['q12c'] = optimal_r2_slr
        q_answers_dict['q12d'] = optimal_b0_slr
        q_answers_dict['q12e'] = optimal_b1_slr
        print(f"[q12a] Optimal value of n (last months from {target_year}): {best_n_slr}")
        print(f"[q12b] Associated adjusted R-squared (R2_adj): {best_adj_r2_slr:.4f}")
        print(f"[q12c] Associated R-squared (R2): {optimal_r2_slr:.4f}")
        print(f"[q12d] Beta0 (intercept): {optimal_b0_slr:.4f}")
        print(f"[q12e] Beta1 (slope for month_ID): {optimal_b1_slr:.4f}")

        # Plot the optimal regression
        optimal_data_plot_slr = df_year.tail(best_n_slr)
        plt.figure(figsize=(10,6))
        plt.scatter(optimal_data_plot_slr['month_ID'], optimal_data_plot_slr['Max_Temperature_Paris'],
                    label=f'Actual Data (last {best_n_slr} months of {target_year})', color='blue')
        pred_y_slr_plot = best_slr_model_results.predict(sm.add_constant(optimal_data_plot_slr['month_ID']))
        plt.plot(optimal_data_plot_slr['month_ID'], pred_y_slr_plot, color='red',
                 label=f'Optimal SLR (n={best_n_slr}, Adj.R2={best_adj_r2_slr:.3f})')
        # Ensure ticks represent actual month names for the plotted data
        plot_month_ids = sorted(optimal_data_plot_slr['month_ID'].unique().astype(int))
        plot_month_names = [month_order_english_list[m_id] for m_id in plot_month_ids if 0 <= m_id < 12]
        plt.xticks(ticks=plot_month_ids, labels=plot_month_names, rotation=45, ha="right")
        plt.xlabel("Month"); plt.ylabel("Max Temperature (°C)")
        plt.title(f"Optimal SLR for Paris {target_year} (Using last {best_n_slr} months)")
        plt.legend(); save_plot(f"optimal_simple_regression_n{best_n_slr}_{target_year}.png")
        print(f"COMMENT Q12: The model using the last {best_n_slr} months of {target_year} data provides the best fit according to adjusted R-squared. Beta1 indicates the average change in temperature for each unit increase in month_ID.")
    else:
        print(f"WARNING: Could not determine optimal SLR model for {target_year}.")
        for q_id_sfx in ['a','b','c','d','e']: q_answers_dict[f'q12{q_id_sfx}'] = "N/A"

    # --- Q13: Prediction for January of the next year ---
    print(f"\n--- Q13: Prediction for January {target_year + 1} (using SLR from {target_year}) ---")
    if best_slr_model_results:
        jan_next_year_month_id = 0 # January is month_ID 0
        # Prepare input for prediction (must match structure used for training)
        X_pred_jan = pd.DataFrame({'const': [1], 'month_ID': [jan_next_year_month_id]})
        predicted_temp_jan_slr = best_slr_model_results.predict(X_pred_jan)[0]
        q_answers_dict['q13a'] = predicted_temp_jan_slr
        print(f"[q13a] Predicted temperature for January {target_year + 1} (SLR): {predicted_temp_jan_slr:.2f}°C")

        actual_temp_jan_pdf = 7.5 # Given in project PDF for Jan 2025
        difference_slr = predicted_temp_jan_slr - actual_temp_jan_pdf
        q_answers_dict['q13b'] = difference_slr
        print(f"[q13b] Difference (Predicted - Actual 7.5°C): {difference_slr:.2f}°C")
        print(f"COMMENT Q13: The difference shows the prediction error for this specific forecast. A smaller absolute difference indicates a better prediction.")
    else:
        print(f"WARNING: No optimal SLR model from Q12 to make predictions for Q13.")
        q_answers_dict['q13a'] = "N/A"; q_answers_dict['q13b'] = "N/A"

    # --- Q14: Null Hypothesis Test for Beta1 (Slope) ---
    print(f"\n--- Q14: Hypothesis Test for Beta1 (Slope) in Optimal SLR Model ({target_year}) ---")
    # H0: Beta1 = 0 (month_ID has no linear effect on temperature)
    # H1: Beta1 != 0 (month_ID has a linear effect)
    if best_slr_model_results:
        if 'month_ID' in best_slr_model_results.pvalues:
            p_value_beta1 = best_slr_model_results.pvalues['month_ID']
            q_answers_dict['q14a'] = p_value_beta1
            print(f"[q14a] P-value for Beta1 (slope coefficient): {p_value_beta1:.4f}")
            alpha = 0.05 # Significance level
            is_significant = p_value_beta1 < alpha
            q_answers_dict['q14b'] = "Yes" if is_significant else "No"
            print(f"[q14b] Is there a significant linear relationship (Beta1 != 0 at alpha=5%)? {'Yes' if is_significant else 'No'}")
            print(f"COMMENT Q14: If the p-value is less than alpha (e.g., 0.05), we reject the null hypothesis and conclude that 'month_ID' is a statistically significant predictor of temperature in this simple linear model.")
        else:
            print("WARNING: 'month_ID' not found in p-values of the best SLR model. Cannot perform test.")
            q_answers_dict['q14a'] = "N/A"; q_answers_dict['q14b'] = "N/A"
    else:
        print("WARNING: No optimal SLR model from Q12 for hypothesis testing in Q14.")
        q_answers_dict['q14a'] = "N/A"; q_answers_dict['q14b'] = "N/A"

    return best_slr_model_results


################################################################################
# 5. MULTIVARIATE LINEAR REGRESSION (MLR) (data2.csv - Paris 2023 & 2024)
################################################################################
# MLR extends SLR by using multiple independent variables (lagged temperatures T-1 to T-12)
# to predict the dependent variable (current month's temperature).
# Model: Temp_i = B0 + B1*Temp_{i-1} + B2*Temp_{i-2} + ... + B12*Temp_{i-12} + error.

def prepare_mlr_data_with_lags(data2_full_for_lags, target_prediction_year):
    """
    Prepares data for MLR by creating lagged temperature features.
    Trains the model on data from `target_prediction_year - 1` (e.g. 2024)
    using lags that can span across `target_prediction_year - 2` (e.g. 2023).

    Args:
        data2_full_for_lags (pandas.DataFrame): Combined, sorted DataFrame of 2023 and 2024 temperatures.
        target_prediction_year (int): The year for which we are training the model to predict (e.g., 2025, meaning training on 2024 data).

    Returns:
        pandas.DataFrame: DataFrame ready for MLR training (usually data for `target_prediction_year - 1` with lags),
                          or an empty DataFrame if not enough data.
    """
    if data2_full_for_lags.empty:
        print("ERROR: Combined data for lags is empty. Cannot prepare MLR data.")
        return pd.DataFrame()

    mlr_df = data2_full_for_lags.copy()
    print("INFO: Creating lagged temperature features (T_lag_1 to T_lag_12 for MLR)...")
    # Lags are previous month's temperatures. T_lag_1 is temp from 1 month ago.
    for lag in range(1, 13):
        mlr_df[f'T_lag_{lag}'] = mlr_df['Max_Temperature_Paris'].shift(lag)

    # Training data for MLR is specified as the 12 months of 2024 in the project PDF.
    # So, if target_prediction_year is 2025, we train on 2024 data.
    training_year_for_mlr = target_prediction_year - 1
    mlr_train_df = mlr_df[mlr_df['Year'] == training_year_for_mlr].dropna() # Drop rows with NaNs from lagging
    # .dropna() is important as early months in the series won't have all 12 lags.
    print(f"INFO: MLR training data (Year {training_year_for_mlr} with complete lags): {len(mlr_train_df)} records.")
    return mlr_train_df

def perform_mlr_analysis_q15_q17(data2_full, data2_year_minus_1, data2_year_current,
                                   month_order_english_list, q_answers_dict,
                                   actual_temps_next_year_pdf):
    """
    Performs Multivariate Linear Regression analysis (Q15-Q17).
    - Q15: Plots temperature evolution for two consecutive years.
    - Prepares data with lagged features.
    - Q16: Finds the optimal combination of up to 12 lagged temperature variables
           that maximizes adjusted R-squared for predicting the current month's temperature.
           Fits the model and reports parameters and F-test.
    - Q17: Predicts temperature for Jan of the next year and iteratively for Feb-Apr.

    Args:
        data2_full (pandas.DataFrame): The preprocessed data2 DataFrame (containing all relevant years).
        data2_year_minus_1 (pandas.DataFrame): Filtered data for e.g. 2023.
        data2_year_current (pandas.DataFrame): Filtered data for e.g. 2024 (used for MLR training).
        month_order_english_list (list): List of English month names.
        q_answers_dict (dict): Dictionary to store answers.
        actual_temps_next_year_pdf (dict): Actual temperatures for future months for comparison.
                                           Example: {'Jan': 7.5, 'Feb': 8.6, ...}
    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: The best MLR model, or None.
    """
    print(f"\n--- Part 4 (MLR): Multivariate Linear Regression (Predicting {data2_year_current['Year'].iloc[0]+1} from {data2_year_current['Year'].iloc[0]} data) ---")

    # --- Q15: Superimpose Temperatures for Two Years ---
    print(f"\n--- Q15: Temperature Evolution ({data2_year_minus_1['Year'].iloc[0]} vs {data2_year_current['Year'].iloc[0]}) ---")
    # Compare temperature trends visually for context.
    year1_label = str(int(data2_year_minus_1['Year'].iloc[0]))
    year2_label = str(int(data2_year_current['Year'].iloc[0]))

    if not data2_year_minus_1.empty and not data2_year_current.empty and \
       data2_year_minus_1['month_ID'].notna().any() and data2_year_current['month_ID'].notna().any():
        plt.figure(figsize=(12, 7))
        plt.plot(data2_year_minus_1['month_ID'], data2_year_minus_1['Max_Temperature_Paris'], marker='o', linestyle='-', label=f'Paris {year1_label}')
        plt.plot(data2_year_current['month_ID'], data2_year_current['Max_Temperature_Paris'], marker='x', linestyle='--', label=f'Paris {year2_label}')
        plt.xticks(ticks=range(12), labels=month_order_english_list, rotation=45, ha="right")
        plt.xlabel("Month"); plt.ylabel("Max Temperature (°C)")
        plt.title(f"Max Temperature in Paris ({year1_label} vs {year2_label})")
        plt.legend(); save_plot(f"paris_temp_{year1_label}_{year2_label}.png")
        q_answers_dict['q15'] = f"Plot 'paris_temp_{year1_label}_{year2_label}.png' generated."
        print(f"COMMENT Q15: Plotting both years helps visualize consistency in seasonal patterns and any year-over-year deviations.")
    else:
        print(f"WARNING: Skipping Q15 plot due to insufficient data for one or both years.")
        q_answers_dict['q15'] = "N/A (Insufficient data)"

    # Prepare combined data (e.g., 2023 and 2024) for creating lags for MLR model trained on 2024 data
    # The MLR model is trained on 2024 data (target_prediction_year - 1)
    # Lags for 2024 data can come from 2023 data (target_prediction_year - 2)
    target_prediction_year = int(data2_year_current['Year'].iloc[0]) + 1 # e.g., 2025
    mlr_training_data = prepare_mlr_data_with_lags(data2_full, target_prediction_year)

    best_mlr_model_results = None
    best_combo_mlr_vars = None
    if mlr_training_data.empty or len(mlr_training_data) < 2: # Need at least 2 records for any regression
        print("WARNING: Not enough training data for MLR after creating lags. Skipping Q16-Q17.")
        for q_id_sfx in ['a', 'b']: q_answers_dict[f'q16{q_id_sfx}'] = "N/A"
        q_answers_dict['q17'] = "N/A"
        return None

    # --- Q16: Optimal MLR Model (Combinations of Lag Variables) ---
    print(f"\n--- Q16: Optimal Multivariate Linear Regression for Paris {target_prediction_year -1} ---")
    # We want to find the best combination of up to 12 lagged temperature variables.
    num_lag_vars_max = 12
    q_answers_dict['q16a'] = 2**num_lag_vars_max - 1 # Number of non-empty subsets
    print(f"[q16a] Number of possible combinations of up to {num_lag_vars_max} lag variables: {q_answers_dict['q16a']}")

    # Identify available lag columns in the training data (some might be all NaN if data is sparse)
    all_possible_lag_cols = [f'T_lag_{i}' for i in range(1, num_lag_vars_max + 1)]
    available_lag_cols = [col for col in all_possible_lag_cols if col in mlr_training_data.columns and not mlr_training_data[col].isnull().all()]

    if not available_lag_cols:
        print("WARNING: No valid lag features available for MLR. Skipping Q16 model fitting.")
        q_answers_dict['q16b'] = "N/A"; q_answers_dict['q17'] = "N/A"
        return None

    print(f"INFO: Finding optimal MLR model using {len(available_lag_cols)} available lag features: {available_lag_cols}")
    y_mlr_train = mlr_training_data['Max_Temperature_Paris']
    best_adj_r2_mlr = -float('inf')

    # Iterate through all numbers of predictors (1 to len(available_lag_cols))
    # and all combinations for that number of predictors.
    for k_predictors in range(1, len(available_lag_cols) + 1):
        for combo_vars in combinations(available_lag_cols, k_predictors):
            X_mlr_combo = mlr_training_data[list(combo_vars)]
            X_mlr_sm_combo = sm.add_constant(X_mlr_combo) # Add intercept
            # Ensure enough observations for the number of parameters (k_predictors + 1 for intercept)
            if X_mlr_sm_combo.shape[0] <= X_mlr_sm_combo.shape[1]: continue # nobs <= n_params
            try:
                model_mlr_iter = sm.OLS(y_mlr_train, X_mlr_sm_combo).fit()
                if model_mlr_iter.rsquared_adj > best_adj_r2_mlr:
                    best_adj_r2_mlr = model_mlr_iter.rsquared_adj
                    best_combo_mlr_vars = list(combo_vars)
                    best_mlr_model_results = model_mlr_iter
            except Exception: continue # Ignore combinations that fail

    if best_mlr_model_results:
        q_answers_dict['q16b'] = len(best_combo_mlr_vars)
        print(f"[q16b] Number of selected variables in optimal MLR: {len(best_combo_mlr_vars)}")
        print(f"INFO: Optimal MLR combination (selected lag variables): {best_combo_mlr_vars}")
        print(f"INFO: Optimal MLR adjusted R-squared: {best_adj_r2_mlr:.4f}")
        print("INFO: Optimal MLR model parameters (coefficients):")
        print(best_mlr_model_results.params)

        # F-test for overall significance of the model
        # H0: All slope coefficients are zero. H1: At least one slope coefficient is non-zero.
        f_pvalue_mlr = best_mlr_model_results.f_pvalue
        alpha_f_test = 0.05
        mlr_overall_sig = f_pvalue_mlr < alpha_f_test
        print(f"INFO: P-value for F-statistic of the optimal MLR model: {f_pvalue_mlr:.4f}")
        print(f"INFO: Is there an overall linear relationship (F-test at alpha={alpha_f_test})? {'Yes' if mlr_overall_sig else 'No'}")
        print("COMMENT Q16: The selected combination of past temperatures provides the best model fit. The F-test tells us if the model, as a whole, is statistically useful for prediction.")
    else:
        print("WARNING: Could not determine optimal MLR model from combinations.")
        q_answers_dict['q16b'] = "N/A"; q_answers_dict['q17'] = "N/A"
        return None

    # --- Q17: Prediction for Next Year (e.g., Jan-Apr 2025) ---
    print(f"\n--- Q17: Predictions for {target_prediction_year} using Optimal MLR Model ---")
    # For Jan 2025 prediction, we need lags up to Dec 2024.
    # These lags come from data2_full (which should contain 2023 and 2024 actuals).
    # The project PDF gives actuals for Jan-Apr 2025 for comparison.

    if not best_mlr_model_results or not best_combo_mlr_vars:
        print(f"WARNING: No optimal MLR model from Q16. Cannot make predictions for Q17.")
        q_answers_dict['q17'] = "N/A (No Q16 MLR model)"
        return best_mlr_model_results # Can still return None if it was None

    # Get the last 12 known temperatures up to the end of the MLR training year (e.g., Dec 2024)
    # These are needed to form the lag features for the first prediction (e.g., Jan 2025).
    # Ensure data2_full has enough history.
    if len(data2_full) < 12:
        print(f"WARNING: Not enough historical data in data2_full ({len(data2_full)} months) to form initial lags for Q17.")
        q_answers_dict['q17'] = "N/A (Insufficient historical data for lags)"
        return best_mlr_model_results

    # Temps up to end of the year *before* target_prediction_year
    temps_up_to_training_end = data2_full[data2_full['Year'] == (target_prediction_year -1)]['Max_Temperature_Paris'].values
    if len(temps_up_to_training_end) < 12 :
         # Try to get last 12 from overall data if specific year is short
         temps_up_to_training_end = data2_full['Max_Temperature_Paris'].tail(12).values
         if len(temps_up_to_training_end) < 12:
            print(f"WARNING: Could not retrieve 12 months of temp data ending {target_prediction_year-1} for Q17 initial lags.")
            q_answers_dict['q17'] = "N/A (Cannot form initial lags)"
            return best_mlr_model_results


    # Iterative prediction for Jan, Feb, Mar, Apr of target_prediction_year
    # We use actual temperatures from PDF for previous months when predicting subsequent ones.
    print(f"INFO: Predicting for Jan-Apr {target_prediction_year} (MLR using actuals for prior months as available):")
    # `temps_history_for_lags` will be updated with actuals as we predict. Start with actuals up to Dec of previous year.
    temps_history_for_lags = list(temps_up_to_training_end) # Should be 12 values ending Dec of (target_prediction_year-1)

    months_to_predict_info = [
        {'name': 'Jan', 'actual_pdf': actual_temps_next_year_pdf.get('Jan')},
        {'name': 'Feb', 'actual_pdf': actual_temps_next_year_pdf.get('Feb')},
        {'name': 'Mar', 'actual_pdf': actual_temps_next_year_pdf.get('Mar')},
        {'name': 'Apr', 'actual_pdf': actual_temps_next_year_pdf.get('Apr')},
    ]

    for month_info in months_to_predict_info:
        month_name_pred = month_info['name']
        actual_temp_pdf_val = month_info['actual_pdf']

        if actual_temp_pdf_val is None:
            print(f"INFO: No actual temperature provided for {month_name_pred} {target_prediction_year}. Skipping its diff calculation.")
            # We might still predict it if we had a way to get previous actuals, but PDF implies we use its given actuals for iterative lags.
            continue

        # Ensure we have enough history (at least 12 months) to form lags
        if len(temps_history_for_lags) < 12:
            print(f"  WARNING: Not enough historical data ({len(temps_history_for_lags)} months) to form 12 lags for {month_name_pred} {target_prediction_year} prediction.")
            if month_name_pred == 'Jan': q_answers_dict['q17'] = "N/A (Lags)" # Specifically for Jan 2025 if it fails
            break # Cannot proceed if lags cannot be formed

        current_lags_source = temps_history_for_lags[-12:] # Use the most recent 12 *actual* temperatures
        features_for_pred_month = {}
        valid_lags_formed = True
        for lag_col_name in best_combo_mlr_vars: # Use the variables from the optimal model
            lag_number = int(lag_col_name.split('_')[-1]) # e.g., T_lag_1 -> 1
            # Lags are relative to the month being predicted.
            # T_lag_1 for Jan is Dec's temp, T_lag_2 is Nov's temp, etc.
            # current_lags_source[11] is most recent (1 month ago), current_lags_source[0] is 12 months ago.
            if (12 - lag_number) >= 0 and (12 - lag_number) < len(current_lags_source):
                features_for_pred_month[lag_col_name] = current_lags_source[12 - lag_number]
            else:
                valid_lags_formed = False; break # Should not happen if current_lags_source has 12 items

        if not valid_lags_formed:
            print(f"  WARNING: Could not form complete feature vector for {month_name_pred} {target_prediction_year} MLR prediction.")
            if month_name_pred == 'Jan': q_answers_dict['q17'] = "N/A (Lags)"
            break

        features_pred_df = pd.DataFrame([features_for_pred_month])
        features_pred_df_sm = sm.add_constant(features_pred_df[best_combo_mlr_vars], has_constant='add') # Ensure correct columns and order

        predicted_temp_mlr = best_mlr_model_results.predict(features_pred_df_sm)[0]
        difference_mlr = predicted_temp_mlr - actual_temp_pdf_val
        print(f"  {month_name_pred} {target_prediction_year}: Predicted (MLR)={predicted_temp_mlr:.2f}°C, Actual (PDF)={actual_temp_pdf_val:.1f}°C, Diff={difference_mlr:.2f}°C")

        if month_name_pred == 'Jan': # Store difference for Jan 2025 as per q17
            q_answers_dict['q17'] = difference_mlr

        # For the *next* month's prediction, add the *actual* temperature of the current month to history.
        temps_history_for_lags.append(actual_temp_pdf_val)

    if 'q17' not in q_answers_dict: # If Jan prediction was skipped
        q_answers_dict['q17'] = "N/A (Prediction for Jan failed or skipped)"

    print(f"COMMENT Q17: The MLR model's predictive performance is evaluated. Using actuals for previous months tests one-step ahead forecast capability. The difference for Jan {target_prediction_year} is specifically requested.")
    return best_mlr_model_results


################################################################################
# 6. GENERATE FINAL ANSWERS CSV
################################################################################

def generate_final_answers_csv(q_answers_dict, question_structure, output_filepath_full):
    """
    Generates a CSV file with quantitative answers based on the project's requirements.

    Args:
        q_answers_dict (dict): Dictionary containing all the answers (q1a, q1b, etc.).
        question_structure (dict): Maps question numbers to lists of their parts (e.g., {1: ['a', 'b']}).
        output_filepath_full (str): The full path to save the CSV file.
    """
    print(f"\n--- Generating Answers CSV: {os.path.basename(output_filepath_full)} ---")
    csv_lines = ["Question_ID,Answer"] # CSV Header

    for q_num in range(1, 18): # Questions 1 to 17 as per PDF
        parts = question_structure.get(q_num, [])
        if not parts: # Single-part question (e.g., q7, q9)
            q_key = f"q{q_num}"
            answer_val = q_answers_dict.get(q_key, 'Not Computed')
            answer_str = f"{answer_val:.4f}" if isinstance(answer_val, float) else str(answer_val)
            csv_lines.append(f"{q_key},{answer_str}")
        else: # Multi-part question
            for part_letter in parts:
                q_key = f"q{q_num}{part_letter}"
                answer_val = q_answers_dict.get(q_key, 'Not Computed')
                answer_str = f"{answer_val:.4f}" if isinstance(answer_val, float) else str(answer_val)
                csv_lines.append(f"{q_key},{answer_str}")

    csv_content_final = "\n".join(csv_lines)
    # print("\n--- Content for answers CSV ---") # Optional: print to console
    # print(csv_content_final)

    try:
        with open(output_filepath_full, "w", encoding='utf-8') as f:
            f.write(csv_content_final)
        print(f"INFO: Answers successfully saved to {output_filepath_full}")
    except Exception as e_write:
        print(f"ERROR: Could not write answers CSV to {output_filepath_full}: {e_write}")


################################################################################
# 7. MAIN EXECUTION ORCHESTRATION
################################################################################

def main():
    """
    Main function to orchestrate the entire data analysis project.
    """
    print("--- Starting Data Science Project Analysis ---")
    create_results_directory_if_not_exists(RESULTS_DIR) # Ensure output directory exists

    # --- Configuration for data1.csv ---
    data1_filename = "data1.csv"
    data1_filepath = os.path.join(DATA_SETS_DIR, data1_filename)
    data1_city_col = 'City'
    data1_rename_map = { # Adjust keys if your CSV has different original names
        'City': 'City', 'Minimum_temperature': 'Temp_min_C',
        'Maximum_temperature': 'Temp_max_C', 'Rainfall': 'Precipitation_mm',
        'Sunshine_duration': 'Ensoleillement_h'
    }
    data1_expected_proc_cols = ['Temp_min_C', 'Temp_max_C', 'Precipitation_mm', 'Ensoleillement_h']

    # --- Configuration for data2.csv ---
    data2_filename = "data2.csv"
    data2_filepath = os.path.join(DATA_SETS_DIR, data2_filename)
    data2_rename_map = { # Adjust keys for your actual CSV column names
        'Maximum_temperature': 'Max_Temperature_Paris'
        # 'MOIS': 'Month', 'ANNEE': 'Year' # Example if French names
    }
    # These are names *after* potential renaming, expected by the script
    data2_required_cols = ['Month', 'Year', 'Max_Temperature_Paris']
    month_order_eng = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

    # --- Load Data ---
    data1 = load_and_preprocess_data1(data1_filepath, data1_city_col, data1_rename_map, data1_expected_proc_cols)
    data2 = load_and_preprocess_data2(data2_filepath, data2_rename_map, data2_required_cols, month_order_eng)

    if data1 is None:
        print("CRITICAL ERROR: data1.csv could not be loaded or preprocessed. Aborting further analysis that depends on it.")
        # Decide if to exit or try to continue with parts that don't depend on data1
    else:
        # --- Part 1: Preliminary Analysis (data1) ---
        data1_cleaned = perform_q1_initial_data_stats(data1.copy(), q_answers) # Use .copy() if df is modified in function

        weather_vars_q2_map = {
            'Temp_min_C': ('lowest minimum temperature', 'highest minimum temperature'),
            'Temp_max_C': ('lowest maximum temperature', 'highest maximum temperature'),
            'Precipitation_mm': ('lowest rainfall', 'highest rainfall'),
            'Ensoleillement_h': ('lowest sunshine duration', 'highest sunshine duration')
        }
        q_labels_q2_list = ['q2a', 'q2b', 'q2c', 'q2d', 'q2e', 'q2f', 'q2g', 'q2h']
        perform_q2_extreme_values(data1_cleaned, weather_vars_q2_map, q_labels_q2_list, q_answers)

        variances_data1 = perform_q3_variances(data1_cleaned, data1_expected_proc_cols, q_answers)
        perform_q4_q5_min_max_variance_stats_and_histograms(data1_cleaned, variances_data1, q_answers)
        perform_q6_variable_correlations(data1_cleaned, data1_expected_proc_cols, q_answers)
        perform_q7_city_correlations(data1_cleaned, data1_expected_proc_cols, q_answers)

        # --- Part 2: PCA (data1) ---
        perform_pca_analysis_q8_q10(data1_cleaned, data1_expected_proc_cols, q_answers)

    if data2 is None:
        print("CRITICAL ERROR: data2.csv could not be loaded or preprocessed. Aborting regression analysis.")
        # Fill Q11-Q17 with N/A if not already handled by individual functions
        for q_num_reg in range(11, 18):
            # This is a fallback, ideally individual functions set their N/As
            if f'q{q_num_reg}' not in q_answers and f'q{q_num_reg}a' not in q_answers :
                 q_answers[f'q{q_num_reg}'] = "N/A (data2 missing)" if q_num_reg in [11,15,17] else None
                 if q_num_reg in [12,13,14,16]: # Multi-part questions
                     parts = question_structure_csv.get(q_num_reg, [])
                     for p in parts: q_answers[f'q{q_num_reg}{p}'] = "N/A (data2 missing)"


    else:
        # --- Part 3: Simple Linear Regression (data2, for 2024) ---
        perform_slr_analysis_q11_q14(data2.copy(), 2024, month_order_eng, q_answers)

        # --- Part 4: Multivariate Linear Regression (data2, for 2023 & 2024) ---
        # For MLR, we need data for 2023 (year_minus_1) and 2024 (year_current for training)
        data2_paris_2023_mlr = data2[
            (data2['Year'] == 2023) & pd.notna(data2['month_ID']) & pd.notna(data2['Max_Temperature_Paris'])
        ].copy().sort_values('month_ID').reset_index(drop=True)

        data2_paris_2024_mlr = data2[
            (data2['Year'] == 2024) & pd.notna(data2['month_ID']) & pd.notna(data2['Max_Temperature_Paris'])
        ].copy().sort_values('month_ID').reset_index(drop=True)

        # Actual temperatures for Jan-Apr 2025 from PDF (for Q17 comparison)
        actual_temps_2025_pdf = {'Jan': 7.5, 'Feb': 8.6, 'Mar': 14.6, 'Apr': 20.0}

        perform_mlr_analysis_q15_q17(data2.copy(), data2_paris_2023_mlr, data2_paris_2024_mlr,
                                      month_order_eng, q_answers, actual_temps_2025_pdf)

    # --- Generate Final Answers CSV ---
    # This structure should match the specific q[ij] keys expected in template.csv
    question_structure_csv = {
        1: ['a', 'b'], 2: [chr(ord('a') + i) for i in range(8)], 3: ['a', 'b', 'c', 'd'],
        4: ['a', 'b', 'c'], 5: ['a', 'b', 'c'], 6: ['a', 'b', 'c'], 7: [], 8: ['a', 'b'],
        9: [], 10: [], 11: [], 12: ['a', 'b', 'c', 'd', 'e'], 13: ['a', 'b'], 14: ['a', 'b'],
        15: [], 16: ['a', 'b'], 17: []
    }
    output_csv_full_path = os.path.join(RESULTS_DIR, "Legrandjacques_Slama_Sartorius.csv")
    generate_final_answers_csv(q_answers, question_structure_csv, output_csv_full_path)

    print("\n--- End of Data Science Project Analysis ---")
    print(f"INFO: All plots and the answers CSV should be in the '{RESULTS_DIR}' directory.")


# This ensures that main() is called only when the script is executed directly
if __name__ == "__main__":
    main()