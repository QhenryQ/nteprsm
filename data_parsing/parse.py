import pandas as pd

# Define the column names
columns = [
    'Entity Name', 'number1', 'number2', 'number3', 'Replications', 'Genetic Color', 
    'Greenup', 'Leaf Texture', 'Traffic designation', 'Wear tolerance', 'Seedling vigor', 
    'January Quality', 'February Quality', 'March Quality', 'April Quality', 'May Quality', 
    'June Quality', 'July Quality', 'August Quality', 'September Quality', 'October Quality', 
    'November Quality', 'December Quality', 'Spring Density', 'Summer_Density', 'Fall_Density', 
    'Spring_ Percent Living Ground Cover1', 'Spring_ Percent Living Ground Cover2', 
    'Summer_ Percent Living Ground Cover1', 'Summer_ Percent Living Ground Cover2',
    'Fall_ Percent Living Ground Cover1', 'Fall_ Percent Living Ground Cover2', 'Frost tolerance', 
    'Winter color', 'Winter kill1', 'Winter kill2', 'Wilting', 'Dormancy', 'Recovery', 
    'Thatch measurements1', 'Thatch measurements2', 'Typhula blight', 'Microdochium patch', 
    'Spring Melting out', 'Fall Melting out', 'Leaf spot', 'Stem rust', 'Dollar spot', 
    'Red thread', 'Brown patch (Warm Temp.)', 'Summer Patch', 'Pythium blight', 'Stripe smut', 
    'Necrotic Ring Spot', 'Crown rust', 'Powdery mildew', 'Anthracnose', 
    'Brown patch (Cool Temp.)', 'Damping-off', 'Fairy ring', 'Gray leaf spot', 'Pink snow mold', 
    'Pink patch', 'Pythium root rot', 'Take-all patch', 'Insect Damage - Please Specify', 
    'September Fall Color Retention', 'October Fall Color Retention', 'November Fall Color Retention', 
    'December Fall Color Retention', 'Seedheads', 'Poa Annua invasion', 'Mowing Quality/Steminess'
]

# Load the data
df = pd.read_excel('test2.xlsx', skiprows=5, usecols=range(0, 73))  # Columns A to BU (0 to 72)
df.columns = columns  # Assign column names

# Ensure 'number1', 'number2', and 'number3' are treated as strings
df['number1'] = df['number1'].astype(str)
df['number2'] = df['number2'].astype(str)
df['number3'] = df['number3'].astype(str)

# Merge number1, number2, and number3 into a new column called 'Entry Code'
df['Entry Code'] = df['number1'] + df['number2'] + df['number3']
df['Entry Code'] = df['Entry Code'].str.zfill(3)

# Drop the original columns if no longer needed
df.drop(['number1', 'number2', 'number3'], axis=1, inplace=True)

# Move 'Entry Code' to the second position
cols = df.columns.tolist()
cols.insert(1, cols.pop(cols.index('Entry Code')))  # Move 'Entry Code' to second position
df = df[cols]

# Read specific cell values for all rows
soil_ph = df.at[0, 'Soil Ph'] = pd.read_excel('test2.xlsx', usecols='BE', nrows=1).iloc[0, 0]
nitrogen_level = df.at[0, 'Nitrogen Level'] = pd.read_excel('test2.xlsx', usecols='BP', nrows=1).iloc[0, 0]
sol_potassium = df.at[0, 'Sol Potassium'] = pd.read_excel('test2.xlsx', usecols='BP', nrows=1, skiprows=1).iloc[0, 0]
soil_phosphorus = df.at[0, 'Soil Phosphorus'] = pd.read_excel('test2.xlsx', usecols='BP', nrows=1, skiprows=2).iloc[0, 0]
irrigation_practiced = df.at[0, 'Irrigation Practiced'] = pd.read_excel('test2.xlsx', usecols='CA', nrows=1).iloc[0, 0]

# Assign these values to all rows in the DataFrame
df['Soil Ph'] = soil_ph
df['Nitrogen Level'] = nitrogen_level
df['Sol Potassium'] = sol_potassium
df['Soil Phosphorus'] = soil_phosphorus
df['Irrigation Practiced'] = irrigation_practiced

# Display the first few rows of the DataFrame
print(df.head())

# Save the modified DataFrame to Excel
df.to_excel('parsed_data.xlsx', index=False)