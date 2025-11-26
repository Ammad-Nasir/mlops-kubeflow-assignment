from src.pipeline_components import load_and_save_boston_csv


if __name__ == "__main__":
    # Generates data/raw/boston_housing.csv for DVC tracking
    csv_path = "data/raw/boston_housing.csv"
    load_and_save_boston_csv(csv_path)
    print(f"Saved dataset to {csv_path}")


