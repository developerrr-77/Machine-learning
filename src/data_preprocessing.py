import hopsworks
import pandas as pd
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

def preprocess_data():
    project = hopsworks.login(project=os.getenv('HOPSWORKS_PROJECT_NAME'), api_key_value=os.getenv('HOPSWORKS_API_KEY'))
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="purchase_orders", version=1)
    df = fg.read()

    # Standardize product names
    df['sku_name'] = df['sku_name'].replace({
        'Syplilis Device': 'Anti SYPHILIS (ARC)',
        'Syring 3cc': 'Syringes 3cc',
        'I.V Set': 'I.V Set',
        # Add more based on analysis
    })

    # Aggregate to weekly demand
    df['date'] = df['order_date'].dt.to_period('W').dt.start_time
    df_weekly = df.groupby(['sku_name', 'date'])['quantity'].sum().reset_index()
    df_weekly.rename(columns={'quantity': 'demand'}, inplace=True)

    # Filter sparse products ( <50 data points)
    product_counts = df_weekly['sku_name'].value_counts()
    valid_products = product_counts[product_counts >= 50].index
    df_weekly = df_weekly[df_weekly['sku_name'].isin(valid_products)]

    logger.info("Data preprocessed")
    return df_weekly