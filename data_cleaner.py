import pandas as pd
import re

def clean_survey_data(input_file, output_file):
    # 1. Load the sheet
    print(f"Loading data from {input_file}...")
    try:
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        else:
            df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # 2. Rename long columns to short keys
    # EDIT THIS DICTIONARY to match your exact survey columns
    rename_mapping = {
        "How many hours of sleep do you get on an average night?": "Sleep_Hours",
        "What is your current Cumulative GPA?": "GPA",
        "On a scale of 1-10, what is your average stress level?": "Stress_Level"
    }
    
    # We rename columns that match our dictionary
    df.rename(columns=rename_mapping, inplace=True)
    print("Columns renamed successfully.")

    # 3. Convert categorical ranges into numerical midpoints
    def convert_sleep_range(val):
        if pd.isna(val):
            return val
            
        val_str = str(val).lower().strip()
        
        # Hardcoded mappings for typical categorical ranges
        if 'less than 5' in val_str or '<5' in val_str:
            return 4.0
        elif '5-6' in val_str:
            return 5.5
        elif '6-7' in val_str:
            return 6.5
        elif '7-8' in val_str:
            return 7.5
        elif '8-9' in val_str:
            return 8.5
        elif 'more than 9' in val_str or '>9' in val_str:
            return 10.0
        else:
            # Fallback: extract numbers and average them
            try:
                nums = re.findall(r'\d+\.?\d*', val_str)
                if nums:
                    return sum(float(x) for x in nums) / len(nums)
                return val
            except:
                return val

    # Apply the conversion if 'Sleep_Hours' is now a column
    if 'Sleep_Hours' in df.columns:
        df['Sleep_Hours'] = df['Sleep_Hours'].apply(convert_sleep_range)
        # Ensure it's numeric so correlation in the GUI works later
        df['Sleep_Hours'] = pd.to_numeric(df['Sleep_Hours'], errors='coerce')
        print("Sleep hour categories converted to numerical midpoints.")
        
    if 'GPA' in df.columns:
        df['GPA'] = pd.to_numeric(df['GPA'], errors='coerce')

    # 4. Save the cleaned data to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"\nCleaned data successfully saved to: {output_file}")

if __name__ == "__main__":
    # -------------------------------------------------------------
    # USAGE INSTRUCTIONS:
    # 1. Update 'input_filename' with your actual file's name.
    # 2. Update the 'rename_mapping' dictionary above.
    # 3. Run this script!
    # -------------------------------------------------------------
    
    input_filename = "raw_survey_data.xlsx"   # change to your file name
    output_filename = "cleaned_data.csv"
    
    print("Please make sure you have updated the file paths and mapping in the script.")
    # Uncomment the following line once you have configured your file names!
    # clean_survey_data(input_filename, output_filename)
