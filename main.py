import pandas as pd                 
import numpy as np                  
from sqlalchemy import create_engine, text 
from dotenv import load_dotenv     
import logging                      
import os                          
import matplotlib.pyplot as plt     
import seaborn as sns             
from datetime import datetime      
from typing import Dict, List, Tuple, 
import json                       

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    
    
    def __init__(self, engine):
        self.engine = engine
        
    def check_nulls(self, table_name: str) -> Dict[str, float]:
        query = f"""
            SELECT 
                column_name,
                (COUNT(*) - COUNT(CASE WHEN {table_name}.* IS NOT NULL THEN 1 END)) 
                * 100.0 / COUNT(*) as null_percentage
            FROM {table_name}
            CROSS JOIN information_schema.columns 
            WHERE table_name = '{table_name}'
            GROUP BY column_name
        """
        try:
            return pd.read_sql(query, self.engine).to_dict('records')
        except Exception as e:
            logger.error(f"Error checking nulls for {table_name}: {str(e)}")
            return {}

class OlistAnalytics:
    
    def __init__(self, engine):
        self.engine = engine
    
    def analyze_sales_patterns(self) -> pd.DataFrame:
        query = """
            WITH monthly_sales AS (
                SELECT 
                    DATE_TRUNC('month', o.order_purchase_timestamp) as month,
                    COUNT(DISTINCT o.order_id) as order_count,
                    SUM(oi.price) as revenue,
                    COUNT(DISTINCT o.customer_id) as customer_count
                FROM orders o
                JOIN order_items oi ON o.order_id = oi.order_id
                WHERE o.order_status = 'delivered'  
                GROUP BY month
            )
            SELECT 
                month,
                order_count,
                revenue,
                customer_count,
                revenue / NULLIF(order_count, 0) as avg_order_value,
                revenue / NULLIF(customer_count, 0) as revenue_per_customer
            FROM monthly_sales
            ORDER BY month
        """
        return pd.read_sql(query, self.engine)
    
    def analyze_customer_behavior(self) -> pd.DataFrame:
        query = """
            WITH customer_metrics AS (
                SELECT 
                    c.customer_id,
                    c.customer_state,
                    COUNT(DISTINCT o.order_id) as order_count,
                    SUM(oi.price) as total_spend,
                    AVG(oi.price) as avg_order_value,
                    MIN(o.order_purchase_timestamp) as first_order,
                    MAX(o.order_purchase_timestamp) as last_order
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                JOIN order_items oi ON o.order_id = oi.order_id
                GROUP BY c.customer_id, c.customer_state
            )
            SELECT 
                customer_state,
                COUNT(*) as customer_count,
                AVG(order_count) as avg_orders_per_customer,
                AVG(total_spend) as avg_customer_value,
                AVG(avg_order_value) as avg_order_value
            FROM customer_metrics
            GROUP BY customer_state
            ORDER BY customer_count DESC
        """
        return pd.read_sql(query, self.engine)
    
    def analyze_product_performance(self) -> pd.DataFrame:
        query = """
            SELECT 
                COALESCE(ct.product_category_name_english, 
                        p.product_category_name) as category,
                COUNT(DISTINCT p.product_id) as product_count,
                COUNT(DISTINCT oi.order_id) as order_count,
                SUM(oi.price) as total_revenue,
                AVG(oi.price) as avg_price,
                AVG(r.review_score) as avg_rating
            FROM products p
            LEFT JOIN product_category_name_translation ct 
                ON p.product_category_name = ct.product_category_name
            JOIN order_items oi ON p.product_id = oi.product_id
            LEFT JOIN order_reviews r ON oi.order_id = r.order_id
            GROUP BY category
            HAVING COUNT(DISTINCT oi.order_id) > 10
            ORDER BY total_revenue DESC
        """
        return pd.read_sql(query, self.engine)
    
    def create_visualizations(self, output_dir: str = 'reports'):
       
        os.makedirs(output_dir, exist_ok=True)
        
        sales_data = self.analyze_sales_patterns()
        plt.figure(figsize=(15, 8))
        plt.plot(sales_data['month'], sales_data['revenue'], marker='o')
        plt.title('Monthly Revenue Trend')
        plt.xlabel('Month')
        plt.ylabel('Revenue (BRL)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sales_trend.png'))
        plt.close()
        
        customer_data = self.analyze_customer_behavior()
        plt.figure(figsize=(12, 8))
        sns.barplot(data=customer_data, 
                   x='customer_state', 
                   y='customer_count')
        plt.title('Customer Distribution by State')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'customer_distribution.png'))
        plt.close()

class ETLPipeline:
    def __init__(self, engine):
        self.engine = engine
        self.data_quality = DataQualityChecker(engine)
    
    def load_csv_to_db(self, file_path: str, table_name: str) -> None:
        try:
            df = pd.read_csv(file_path)
            
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {file_path} into {table_name}")
            
            null_checks = self.data_quality.check_nulls(table_name)
            if null_checks:
                logger.info(f"Null check results for {table_name}: {json.dumps(null_checks, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

def main():
    try:
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        
        etl = ETLPipeline(engine)
        analytics = OlistAnalytics(engine)
        
        data_files = {
            'customers': 'olist_customers_dataset.csv',
            'orders': 'olist_orders_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }
        
        for table, file in data_files.items():
            etl.load_csv_to_db(os.path.join('data', file), table)
        
        print("\nAnalyzing sales patterns...")
        sales_metrics = analytics.analyze_sales_patterns()
        print(sales_metrics.tail().to_string())
        
        print("\nAnalyzing customer behavior...")
        customer_metrics = analytics.analyze_customer_behavior()
        print(customer_metrics.head().to_string())
        
        print("\nAnalyzing product performance...")
        product_metrics = analytics.analyze_product_performance()
        print(product_metrics.head().to_string())
                print("\nCreating visualizations...")
        analytics.create_visualizations()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
