import os
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
import pdfplumber

class DataIngestion:
    SUPPORTED_FORMATS = ["csv", "xlsx", "xls"]

    def __init__(self, raw_folder: str = "raw", data_folder: str = "Data"):
        self.raw_folder = raw_folder
        self.data_folder = data_folder

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.data_folder, exist_ok=True)

    def load_file(self, file_path: str):
        ext = file_path.split(".")[-1].lower()

        if ext == "csv":
            return pd.read_csv(file_path)

        if ext in ["xlsx", "xls"]:
            return pd.read_excel(file_path)

        raise ValueError(f"Unsupported file type for ML ingestion: {ext}")

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # 👉 Drop 'Borderlands' column if present
        if "Borderlands" in df.columns:
            df = df.drop(columns=["Borderlands"])

        # 👉 Fix duplicate IDs
        if "ID" in df.columns:
            df = df.drop_duplicates(subset=["ID"], keep="first")

        # 👉 Sentiments mapping
        sentiment_map = {
            "Negative": 0,
            "Positive": 1,
            "Neutral": 2
        }

        if "Sentiments" in df.columns:
            df["Sentiments"] = df["Sentiments"].map(sentiment_map)

        return df

    def ingest_data(self):
        files = os.listdir(self.raw_folder)

        for file_name in files:
            full_path = os.path.join(self.raw_folder, file_name)

            if not os.path.isfile(full_path):
                continue

            extension = file_name.split(".")[-1].lower()

            if extension not in self.SUPPORTED_FORMATS:
                print(f"Skipping unsupported file: {file_name}")
                continue

            print(f"Processing file: {file_name}")

            try:
                df = self.load_file(full_path)

                # 👉 Preprocess (drop columns, fix IDs, map sentiments)
                df = self.preprocess_dataframe(df)

                # 👉 Train-test split
                train_df, test_df = train_test_split(
                    df, test_size=0.2, random_state=42
                )

                # Save files in Data/ folder ONLY
                train_path = os.path.join(self.data_folder, f"{file_name}_train.csv")
                test_path = os.path.join(self.data_folder, f"{file_name}_test.csv")

                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

                print(f"✔ Train saved: {train_path}")
                print(f"✔ Test saved:  {test_path}")
                print("✔ Raw file kept untouched.\n")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                continue


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.ingest_data()
    print("\nIngestion & preprocessing completed successfully.")
