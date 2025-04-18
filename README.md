[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=19227404)
# Assignment: Debugging and Big Data Analysis 🐛📊

---

## Overview

This assignment has two parts:

1. **Debugging Python code**
2. **Analyzing large health data**

---

## Part 1: Debugging

### Tasks

- Fix the provided buggy scripts:
  - `1_patient_data_cleaner.py` (cleans and filters patient records)
  - `2_med_dosage_calculator.py` (calculates medication dosages)
- The first script has labeled bugs to help you get started
- The second script has more subtle bugs that require using a debugger to find
- Use **any debugging method you prefer**:
  - Print statements
  - `pdb`
  - VS Code debugger
  - Other tools
- Pass all provided **pytest** tests:
  - `test_patient_data_cleaner.py`
  - `test_med_dosage_calculator.py`
- Add comments explaining:
  - What was wrong (use the comment format: `# BUG: description of the bug`)
  - How you fixed it (use the comment format: `# FIX: description of the fix`)
- **Important for autograding**: Do not change function names or return types

### Requirements

- All tests must pass
- Clear explanations in comments
- Clean, readable code

---

## Part 2: Big Data Analysis

### Tasks

1. **Patient Cohort Analysis**:
   - Complete the `3_cohort_analysis.py` script to analyze patient data
   - Use polars' lazy evaluation and streaming capabilities
   - Convert the input CSV to a parquet file for efficient processing
   - Filter out BMI outliers (values < 10 or > 60)
   - Group patients by BMI ranges and calculate statistics
   - Print summary statistics of the cohorts

2. **Documentation**:
   - Write a brief report in `analysis.md`:
     - Explain your analysis approach
     - Discuss any patterns or insights found
     - Describe how you used polars' features for efficiency

### File Paths and Requirements

- Input file: `patients_large.csv` (generated by `generate_large_health_data.py`)
- Parquet file: `patients_large.parquet` (you must create this)

### Sample Data

The input data has the following structure:

| Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | Outcome |
|-------------|---------|---------------|---------------|---------|-----|-------------------------|-----|---------|
| 6 | 148 | 72 | 35 | 0 | 33.6 | 0.627 | 50 | 1 |
| 1 | 85 | 66 | 29 | 0 | 26.6 | 0.351 | 31 | 0 |
| 8 | 183 | 64 | 0 | 0 | 23.3 | 0.672 | 32 | 1 |

The `generate_large_health_data.py` script will create a larger version of this dataset with:

- 5 million rows
- Added noise to Age and Glucose values
- A diagnosis column based on the Outcome

### BMI Ranges

Use these standard WHO BMI ranges for cohort analysis, with outlier handling:

- Filter out BMI values < 10 or > 60 (these are likely data errors)
- Underweight: 10 ≤ BMI < 18.5
- Normal: 18.5 ≤ BMI < 25
- Overweight: 25 ≤ BMI < 30
- Obese: 30 ≤ BMI ≤ 60

Note: The ranges use left-closed intervals [a, b) except for the last range which is [a, b].

### Expected Output Format

The final output should be a DataFrame with the following columns:

- `bmi_range`: The BMI range (e.g., "Underweight", "Normal", "Overweight", "Obese")
- `avg_glucose`: Mean glucose level by BMI range
- `patient_count`: Number of patients by BMI range
- `avg_age`: Mean age by BMI range

### Autograding Requirements

For successful autograding:

1. Do not rename any functions or files
2. Follow the exact file naming conventions specified
3. Ensure your code runs without errors
4. Make sure all output has the exact column names specified
5. Use the required comment formats for bug documentation

---

## Submission Checklist

- Fixed Python scripts with bug documentation
- All pytest tests passing
- Completed `cohort_analysis.py` script
- `analysis.md` with documentation

---

## Notes

<!--
The debugging portion teaches systematic debugging with increasing difficulty.
The big data portion focuses on real-world health data analysis scenarios.
-->