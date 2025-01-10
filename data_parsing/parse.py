import pandas as pd
import json
from collections import deque

# Load the JSON file and convert lists back to deques
with open('entity_data.json', 'r') as file:
    loaded_data = json.load(file)
    entity_data = {key: deque(value) for key, value in loaded_data.items()}

# Define the column names
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

# Load the data from Excel
file_path = './data/2019zoysiatrial/2019 Zoysia NTEP Data_Dallas Data_2021 submission.xlsx'
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

# Iterate through the DataFrame and match with the queue
for index, row in df.iterrows():
    # Get the entity_name
    entity_name = row['entity_name']
 
    # Check if the entity_name exists in the entity_data dictionary
    if entity_name in entity_data and len(entity_data[entity_name]) > 0:
        # Dequeue one object from the corresponding queue
        dequeued_item = entity_data[entity_name].popleft()
        print(dequeued_item)
        
        # Assign values directly to the DataFrame
        df.at[index, 'row'] = dequeued_item['row']
        df.at[index, 'column'] = dequeued_item['column']

# Display the first few rows of the modified DataFrame
print(df[['row', 'column']].head())
df['row'] = df['row'].astype(str).str.zfill(2)
df['column'] = df['column'].astype(str).str.zfill(2)
df['plt_id'] = df['row'] + df['column']
# Save the modified DataFrame to Excel
df.to_csv('2021parsed_data11.csv', index=False)