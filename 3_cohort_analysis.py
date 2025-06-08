import polars as pl
from pathlib import Path

def analyze_patient_cohorts(input_file: str) -> pl.DataFrame:
    """
    Analyze patient cohorts based on BMI ranges.
    
    Args:
        input_file: Path to the input CSV file
        
    Returns:
        DataFrame containing cohort analysis results with columns:
        - bmi_range: The BMI range (e.g., "Underweight", "Normal", "Overweight", "Obese")
        - avg_glucose: Mean glucose level by BMI range
        - patient_count: Number of patients by BMI range
        - avg_age: Mean age by BMI range
    """
    # Convert CSV to Parquet for efficient processing
    csv_path = Path(input_file)
    parquet_path = csv_path.with_suffix('.parquet')    
    
    # Create a lazy query to analyze cohorts
    cohort_results = (
        pl.scan_parquet(parquet_path)
        .filter((pl.col("BMI") >= 10) & (pl.col("BMI") <= 60))
        .select(["BMI", "Glucose", "Age"])
        .with_columns(
            pl.when(pl.col("BMI") < 18.5).then(pl.lit("Underweight"))
             .when(pl.col("BMI") < 25).then(pl.lit("Normal"))
             .when(pl.col("BMI") < 30).then(pl.lit("Overweight"))
             .otherwise(pl.lit("Obese"))
             .alias("bmi_range")
        )
        .filter(pl.col("bmi_range").is_not_null())
        .group_by("bmi_range")
        .agg([
            pl.col("Glucose").mean().alias("avg_glucose"),
            pl.len().alias("patient_count"),
            pl.col("Age").mean().alias("avg_age")
        ])
        .collect()
        .sort("bmi_range")
    )
    
    return cohort_results

def main():
    # Input file
    input_file = "patients_large.csv"
    
    # Run analysis
    results = analyze_patient_cohorts(input_file)
    
    # Print summary statistics
    print("\nCohort Analysis Summary:")
    print(results)

if __name__ == "__main__":
    main() 