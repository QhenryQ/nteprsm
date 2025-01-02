import pandas as pd

# Define the column names with lowercase and underscores
columns = [
    'entity_name', 'number1', 'number2', 'number3', 'replications', 'genetic_color', 
    'greenup', 'leaf_texture', 'traffic_designation', 'wear_tolerance', 'seedling_vigor', 
    'january_quality', 'february_quality', 'march_quality', 'april_quality', 'may_quality', 
    'june_quality', 'july_quality', 'august_quality', 'september_quality', 'october_quality', 
    'november_quality', 'december_quality', 'spring_density', 'summer_density', 'fall_density', 
    'spring_percent_living_ground_cover1', 'spring_percent_living_ground_cover2', 
    'summer_percent_living_ground_cover1', 'summer_percent_living_ground_cover2',
    'fall_percent_living_ground_cover1', 'fall_percent_living_ground_cover2', 'frost_tolerance', 
    'winter_color', 'winter_kill1', 'winter_kill2', 'wilting', 'dormancy', 'recovery', 
    'thatch_measurements1', 'thatch_measurements2', 'typhula_blight', 'microdochium_patch', 
    'spring_melting_out', 'fall_melting_out', 'leaf_spot', 'stem_rust', 'dollar_spot', 
    'red_thread', 'brown_patch_warm_temp', 'summer_patch', 'pythium_blight', 'stripe_smut', 
    'necrotic_ring_spot', 'crown_rust', 'powdery_mildew', 'anthracnose', 
    'brown_patch_cool_temp', 'damping_off', 'fairy_ring', 'gray_leaf_spot', 'pink_snow_mold', 
    'pink_patch', 'pythium_root_rot', 'take_all_patch', 'insect_damage_please_specify', 
    'september_fall_color_retention', 'october_fall_color_retention', 'november_fall_color_retention', 
    'december_fall_color_retention', 'seedheads', 'poa_annua_invasion', 'mowing_quality_steminess',
]
file_path = './data/2023submission.xlsx'

# Load the data with openpyxl
df = pd.read_excel(file_path, skiprows=5, usecols=range(0, 73), engine='openpyxl')  # Columns A to BU (0 to 72)
df.columns = columns  # Assign column names

# Ensure 'number1', 'number2', and 'number3' are treated as strings
df['number1'] = df['number1'].astype(str)
df['number2'] = df['number2'].astype(str)
df['number3'] = df['number3'].astype(str)

# Merge number1, number2, and number3 into a new column called 'entry_code'
df['entry_code'] = df['number1'] + df['number2'] + df['number3']
df['entry_code'] = df['entry_code'].str.zfill(3)

# Drop the original columns if no longer needed
df.drop(['number1', 'number2', 'number3'], axis=1, inplace=True)

# Move 'entry_code' to the second position
cols = df.columns.tolist()
cols.insert(1, cols.pop(cols.index('entry_code')))  # Move 'entry_code' to second position
df = df[cols]

# Initialize the new blank columns
df['row'] = ""
df['column'] = ""
df['plt_id'] = ""
df['comp'] = ""
df['rater'] = ""

# Display the first few rows of the DataFrame
print(df.head())

# Save the modified DataFrame to Excel
df.to_excel('2023parsed_data.xlsx', index=False)