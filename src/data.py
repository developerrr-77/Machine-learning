import os
from dotenv import load_dotenv
import pandas as pd
import hopsworks
import logging
from retrying import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def retry_if_exception(exception):
    return isinstance(exception, Exception)

@retry(retry_on_exception=retry_if_exception, stop_max_attempt_number=3, wait_fixed=5000)
def read_feature_group(fs):
    fg = fs.get_feature_group(name="purchase_orders", version=1)
    return fg.read()

def ingest_data(file_path):
    try:
        # Connect to Hopsworks
        project = hopsworks.login(
        
            api_key_value=os.getenv('HOPSWORKS_API_KEY')
        )
        fs = project.get_feature_store()

        # Read data with retries
        df = read_feature_group(fs)
        logger.info("Data fetched from Hopsworks Feature Group")

        # Standardize product names
        df['sku_name'] = df['sku_name'].replace({
            'Syplilis Device': 'Anti SYPHILIS (ARC)',
            'Syring 3cc': 'Syringes 3cc',
            'I.V Set': 'I.V. Set',
            'Syringes 5CC': 'Syringes 5cc',
            'Syringes 10CC': 'Syringes 10cc'
        })

        # Aggregate to weekly demand
        df['date'] = pd.to_datetime(df['order_date']).dt.to_period('W').dt.start_time
        df_weekly = df.groupby(['sku_name', 'date'])['quantity'].sum().reset_index()
        df_weekly.rename(columns={'quantity': 'demand'}, inplace=True)

        # Filter sparse products
        product_counts = df_weekly['sku_name'].value_counts()
        valid_products = product_counts[product_counts >= 50].index
        df_weekly = df_weekly[df_weekly['sku_name'].isin(valid_products)]
        logger.info(f"Preprocessed data: {df_weekly.shape[0]} rows, filtered to {len(valid_products)} products")

        return df_weekly

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise