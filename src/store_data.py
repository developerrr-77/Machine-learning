import hopsworks
import pandas as pd
import logging
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY" )
def store_transformed_to_feature_store(filename: str, feature_group_name: str, version: int = 1):
    """
    Stores a transformed CSV/Excel file into Hopsworks Feature Store.
    """
    try:
        logging.info("Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value = api_key)  # Will ask for API key if not set in env

        fs = project.get_feature_store()

        # Load transformed file
        root_dir = Path(__file__).resolve().parent.parent
        file_path = root_dir / "data" / "transformed" / filename

        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        logging.info(f"Loaded transformed file: {file_path}")

        # Create Feature Group (or get existing)
        fg = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=version,
            primary_key=["order_date", "product_id"],
            description="Transformed sales data with missing dates filled"
        )

        # Insert data into feature group
        fg.insert(df, write_options={"wait_for_job": True})
        logging.info(f"Data inserted into Feature Store: {feature_group_name} v{version}")

    except Exception as e:
        logging.error(f"Error storing to Feature Store: {e}", exc_info=True)

if __name__ == "__main__":
    # Example usage
    store_transformed_to_feature_store(
        filename="transformed_data.xlsx",
        feature_group_name="demand_records",
        version=1
    )
