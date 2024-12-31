# README for Data Processing Program

## Overview

This Python program processes agricultural data from the NTEP Report and outputs a tabular version of the data to a new Excel file. The program utilizes the `pandas` library for data manipulation and is designed to parse specific data formats as required for analysis.

## Features

- Reads and processes data from specified columns in an Excel file.
- Merges specific columns to create a new `Entry Code`.
- Extracts specific soil and irrigation parameters to add to the dataset.
- Saves the cleaned and modified data to a new Excel file.

## Data Format

The program expects the input Excel file (`test2.xlsx`) to have the following structure:

- The relevant data starts from the 6th row (indexed as 5).
- It reads columns A to BU (0 to 72) and assigns predefined column names to the DataFrame.
- The program also reads specific values from cells `BE1`, `BD1`, `BP1`, and `CA1` for additional parameters.

## Parameters Extracted

The following parameters are extracted and included in the output:

- **Soil pH** (from cell `BE3`)
- **Nitrogen Level** (from cell `BP3`)
- **Soluble Potassium** (from cell `BP2`)
- **Soil Phosphorus** (from cell `BP1`)
- **Irrigation Practiced** (from cell `CA3`)

## Output

The processed data is saved to `output7.xlsx`. This output includes:

- The original data with added columns for the extracted parameters.
- A new `Entry Code` column created from the concatenation of `number1`, `number2`, and `number3`.

## Manual Input Note

Due to inconsistencies in blank columns within the provided report format, some columns may require manual input or verification after the initial processing. It is essential to review the output data for any discrepancies or required adjustments.

## Requirements

- Python 3.x
- pandas library

You can install the required library using pip:

```bash
pip install pandas openpyxl
```

## Usage

1. Place the report in the same directory as the script.
2. Run the script using Python.
3. Check the generated `output7.xlsx` for the processed data.

## Conclusion

This program serves as a useful tool for parsing and processing agricultural data efficiently. However, users should be aware of potential inconsistencies in the data and be prepared to make necessary manual adjustments.