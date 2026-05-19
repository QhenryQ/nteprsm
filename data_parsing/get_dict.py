import pandas as pd
from collections import deque
import json
# Load the Excel file
df = pd.read_excel('./parsed_data/2019parsed_data.xlsx')

# Initialize a dictionary to store the data for each entity using a deque
entity_data = {}

# Group the DataFrame by 'entity_name'
for entity, group in df.groupby('entity_name'):
    # Use deque to store rows and columns for each entity
    entity_data[entity] = deque(group[['row', 'column']].to_dict(orient='records'))

# Example output: You now have a dictionary where each key is an entity name, 
# and the value is a deque containing the rows and columns as dictionaries
print(entity_data)
with open('entity_data.json', 'w') as file:
    # Convert deques to lists for JSON serialization
    json.dump({key: list(value) for key, value in entity_data.items()}, file, indent=4)
