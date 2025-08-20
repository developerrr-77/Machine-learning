import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def read_file(filename: str):
    """
    Reads an Excel or CSV file from ../data/raw folder.
    Returns a DataFrame if found, else None.
    """
    try:
        root_dir = Path(__file__).resolve().parent.parent
        file_path = root_dir / "data" / "raw" / filename

        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            return None

        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            logging.error(f"Unsupported file format: {file_path.suffix}")
            return None

        logging.info(f"File loaded successfully: {file_path}")
        print(df.head())
        return df

    except Exception as e:
        logging.error(f"Error reading file {filename}: {e}", exc_info=True)
        return None


def clean_data(df: pd.DataFrame):
    """
    Cleans the given DataFrame by removing dirt and fixing improper values.
    """
    if df is None or df.empty:
        logging.error("No data provided for cleaning.")
        return None

    try:
        logging.info("Cleaning data...")
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df.dropna(how="all", inplace=True)
        dirt_values = ["N/A", "n/a", "-", "null", "NULL", ""]
        df.replace(dirt_values, pd.NA, inplace=True)
        logging.info("Data cleaned successfully.")
        return df

    except Exception as e:
        logging.error(f"Error cleaning data: {e}", exc_info=True)
        return None


def transform_data(df: pd.DataFrame, filename: str):
    """
    Transforms the DataFrame:
    - Renames columns
    - Creates missing dates globally with quantity=0
    - Sorts by date descending
    - Saves transformed file in ../data/transformed
    """
    if df is None or df.empty:
        logging.error("No data provided for transformation.")
        return None

    try:
        logging.info("Transforming data...")

        # Rename columns
        rename_map = {
            "PONumber": "order_number",
            "OrderDate": "order_date",
            "SkuName": "product_id",
            "SupplierName": "supplier",
            "NoOfPItems": "quantity"
        }
        df.rename(columns=rename_map, inplace=True)

        # Ensure datetime for order_date
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        # Fill missing quantity with 0
        df["quantity"] = df["quantity"].fillna(0)

        # Create full global date range
        min_date = df["order_date"].min()
        max_date = df["order_date"].max()
        all_dates = pd.DataFrame({"order_date": pd.date_range(min_date, max_date)})

        # Merge existing data with all dates
        df = pd.merge(all_dates, df, on="order_date", how="left")

        # Fill NaNs for missing rows
        df["quantity"] = df["quantity"].fillna(0)
        df["product_id"] = df["product_id"].fillna("No Product")
        df["order_number"] = df["order_number"].fillna("No Order")
        df["supplier"] = df["supplier"].fillna("No Supplier")

        # Sort by date descending
        df.sort_values(by="order_date", ascending=False, inplace=True)

        logging.info("Transformation complete.")
        print(df.head())

        # Save transformed file
        root_dir = Path(__file__).resolve().parent.parent
        save_dir = root_dir / "data" / "transformed"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / filename
        if save_path.suffix.lower() in [".xlsx", ".xls"]:
            df.to_excel(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)

        logging.info(f"Transformed file saved to: {save_path}")
        return df

    except Exception as e:
        logging.error(f"Error transforming data: {e}", exc_info=True)
        return None
