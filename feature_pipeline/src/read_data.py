import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def fetch_data():
    try:
        # Get source file path from environment variable
        source_file = Path(os.getenv("SOURCE_FILE_PATH", "")).resolve()
        
        if not source_file.exists():
            logging.error(f"Source file not found: {source_file}")
            return

        # Define save directory
        root_dir = Path(__file__).resolve().parent.parent
        save_dir = root_dir / "data" / "raw"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / "SundasPoDetail.xlsx"

        # If file already exists, skip
        if save_path.exists():
            logging.info(f"File already exists, skipping: {save_path}")
            return

        # Read and save file
        logging.info(f"Reading source file: {source_file}")
        df = pd.read_excel(source_file)
        
        df.to_excel(save_path, index=False)
        logging.info(f"File saved to: {save_path}")

    except Exception as e:
        logging.error(f"Error while fetching data: {e}", exc_info=True)


if __name__ == "__main__":
    fetch_data()
