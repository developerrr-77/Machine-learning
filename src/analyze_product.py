import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def analyze_product_id_positive_quantity(filename: str):
    """
    Reads transformed file and prints frequency of product_ids 
    where quantity > 0.
    """
    try:
        # Path to transformed folder
        root_dir = Path(__file__).resolve().parent.parent
        file_path = root_dir / "data" / "transformed" / filename

        if not file_path.exists():
            logging.error(f"Transformed file not found: {file_path}")
            return

        # Load file
        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        else:
            logging.error(f"Unsupported file format: {file_path.suffix}")
            return

        # Check required columns
        if "product_id" not in df.columns or "quantity" not in df.columns:
            logging.error("Required columns ['product_id', 'quantity'] not found.")
            return

        # Filter for rows with quantity > 0
        df_positive = df[df["quantity"] > 0]

        # Frequency table
        freq_table = (
            df_positive["product_id"]
            .value_counts()
            .reset_index()
        )
        freq_table.columns = ["product_id", "record_count"]

        print("\nFrequency Table (quantity > 0):\n", freq_table)

    except Exception as e:
        logging.error(f"Error analyzing product_id with positive quantity: {e}", exc_info=True)


if __name__ == "__main__":
    analyze_product_id_positive_quantity("SundasPoDetail_transformed.xlsx")
