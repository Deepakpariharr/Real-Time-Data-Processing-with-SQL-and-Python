import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Handles data quality validation and reporting."""
    
    def __init__(self, engine):
        self.engine = engine
        
    def check_nulls(self, table_name: str) -> Dict[str, float]:
        """Calculate null percentage for each column in a table."""
        query = f"""
            SELECT 
                column_name,
                (COUNT(*) - COUNT(CASE WHEN {table_name}.* IS NOT NULL THEN 1 END)) * 100.0 / COUNT(*) as null_percentage
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
            
    def check_duplicates(self, table_name: str, key_columns: List[str]) -> int:
        """Check for duplicate records based on key columns."""
        key_cols = ", ".join(key_columns)
        query = f"""
            SELECT COUNT(*) - COUNT(DISTINCT ({key_cols})) as duplicate_count
            FROM {table_name}
        """
        try:
            return pd.read_sql(query, self.engine).iloc[0, 0]
        except Exception as e:
            logger.error(f"Error checking duplicates for {table_name}: {str(e)}")
            return 0
            
    def validate_date_ranges(self, table_name: str, date_column: str) -> Dict[str, datetime]:
        """Validate date ranges in a given column."""
        query = f"""
            SELECT 
                MIN({date_column}) as min_date,
                MAX({date_column}) as max_date
            FROM {table_name}
        """
        try:
            return pd.read_sql(query, self.engine).to_dict('records')[0]
        except Exception as e:
            logger.error(f"Error validating dates for {table_name}.{date_column}: {str(e)}")
            return {}

class OlistAnalytics:
    """Handles advanced analytics and visualizations for Olist data."""
    
    def __init__(self, engine):
        self.engine = engine
        
    def get_time_based_metrics(self) -> pd.DataFrame:
        """Calculate monthly sales metrics."""
        query = """
            WITH monthly_metrics AS (
                SELECT 
                    DATE_TRUNC('month', o.order_purchase_timestamp) as month,
                    COUNT(DISTINCT o.order_id) as total_orders,
                    COUNT(DISTINCT o.customer_id) as unique_customers,
                    SUM(oi.price) as revenue,
                    SUM(oi.freight_value) as freight_cost,
                    AVG(EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_purchase_timestamp))/86400) as avg_delivery_days
                FROM orders o
                JOIN order_items oi ON o.order_id = oi.order_id
                WHERE o.order_status = 'delivered'
                GROUP BY month
                ORDER BY month
            )
            SELECT 
                month,
                total_orders,
                unique_customers,
                revenue,
                freight_cost,
                avg_delivery_days,
                revenue/NULLIF(total_orders, 0) as avg_order_value,
                LAG(revenue) OVER (ORDER BY month) as prev_month_revenue,
                (revenue - LAG(revenue) OVER (ORDER BY month)) / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100 as revenue_growth
            FROM monthly_metrics
        """
        return pd.read_sql(query, self.engine)
        
    def get_product_category_analysis(self) -> pd.DataFrame:
        """Analyze performance by product category."""
        try:
            # First verify the table exists
            table_check_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'category_translation'
                )
            """
            table_exists = pd.read_sql(table_check_query, self.engine).iloc[0, 0]
            
            if table_exists:
                query = """
                    SELECT 
                        COALESCE(ct.product_category_name_english, p.product_category_name) as category,
                        COUNT(DISTINCT oi.order_id) as total_orders,
                        COUNT(DISTINCT oi.product_id) as unique_products,
                        SUM(oi.price) as revenue,
                        AVG(r.review_score) as avg_rating,
                        SUM(oi.price) / COUNT(DISTINCT oi.order_id) as avg_order_value
                    FROM order_items oi
                    JOIN products p ON oi.product_id = p.product_id
                    LEFT JOIN category_translation ct 
                        ON p.product_category_name = ct.product_category_name
                    LEFT JOIN order_reviews r ON oi.order_id = r.order_id
                    GROUP BY category
                    HAVING COUNT(DISTINCT oi.order_id) > 100
                    ORDER BY revenue DESC
                """
            else:
                query = """
                    SELECT 
                        p.product_category_name as category,
                        COUNT(DISTINCT oi.order_id) as total_orders,
                        COUNT(DISTINCT oi.product_id) as unique_products,
                        SUM(oi.price) as revenue,
                        AVG(r.review_score) as avg_rating,
                        SUM(oi.price) / COUNT(DISTINCT oi.order_id) as avg_order_value
                    FROM order_items oi
                    JOIN products p ON oi.product_id = p.product_id
                    LEFT JOIN order_reviews r ON oi.order_id = r.order_id
                    GROUP BY category
                    HAVING COUNT(DISTINCT oi.order_id) > 100
                    ORDER BY revenue DESC
                """

            return pd.read_sql(query, self.engine)

        except Exception as e:
            logger.error(f"Error in get_product_category_analysis: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on failure
        
    def get_seller_performance(self) -> pd.DataFrame:
        """Analyze seller performance metrics."""
        query = """
            WITH seller_metrics AS (
                SELECT 
                    s.seller_id,
                    s.seller_state,
                    COUNT(DISTINCT oi.order_id) as total_orders,
                    COUNT(DISTINCT oi.product_id) as unique_products,
                    SUM(oi.price) as revenue,
                    AVG(r.review_score) as avg_rating,
                    AVG(EXTRACT(EPOCH FROM (o.order_delivered_carrier_date - o.order_purchase_timestamp))/3600) as avg_shipping_time
                FROM sellers s
                JOIN order_items oi ON s.seller_id = oi.seller_id
                JOIN orders o ON oi.order_id = o.order_id
                LEFT JOIN order_reviews r ON oi.order_id = r.order_id
                GROUP BY s.seller_id, s.seller_state
            )
            SELECT 
                *,
                NTILE(5) OVER (ORDER BY revenue DESC) as revenue_quintile,
                NTILE(5) OVER (ORDER BY avg_rating DESC) as rating_quintile
            FROM seller_metrics
            WHERE total_orders >= 10
        """
        return pd.read_sql(query, self.engine)
        
    def create_visualizations(self, output_dir: str = 'reports'):
        """Generate and save visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Time-based metrics visualization
        time_metrics = self.get_time_based_metrics()
        plt.figure(figsize=(15, 8))
        plt.plot(time_metrics['month'], time_metrics['revenue'], marker='o')
        plt.title('Monthly Revenue Trend')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'revenue_trend.png'))
        plt.close()
        
        # Category performance visualization
        cat_metrics = self.get_product_category_analysis()
        plt.figure(figsize=(15, 8))
        sns.scatterplot(data=cat_metrics.head(20), 
                       x='revenue', 
                       y='avg_rating', 
                       size='total_orders',
                       alpha=0.6)
        plt.title('Category Performance: Revenue vs Rating')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_performance.png'))
        plt.close()
        
        # Seller performance distribution
        seller_metrics = self.get_seller_performance()
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=seller_metrics, x='revenue_quintile', y='avg_rating')
        plt.title('Seller Rating Distribution by Revenue Quintile')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seller_performance.png'))
        plt.close()

class ETLPipeline:
    """Handles data loading and transformation."""
    
    def __init__(self, engine):
        self.engine = engine
        self.data_quality = DataQualityChecker(engine)
        
    def load_csv_to_db(self, file_path: str, table_name: str) -> None:
        """Load CSV file into database with error handling and logging."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return
            
            # Load and clean data
            df = pd.read_csv(file_path)
            
            # Basic data cleaning
            for column in df.columns:
                if df[column].dtype == 'object':
                    df[column] = df[column].fillna('')  # Fill NaN in text columns with empty string
                
            # Convert to database-friendly types
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'timestamp' in col.lower()]
            for date_col in date_columns:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                except Exception as e:
                    logger.warning(f"Could not convert {date_col} to datetime: {str(e)}")
            
            # Load to database
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {file_path} into {table_name}")
            
            # Perform data quality checks
            null_checks = self.data_quality.check_nulls(table_name)
            if null_checks:
                logger.info(f"Null check results for {table_name}: {json.dumps(null_checks, indent=2)}")
            
        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {file_path}")
        except pd.errors.ParserError:
            logger.error(f"Error parsing CSV file: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
            
    def create_indexes(self) -> None:
        """Create necessary indexes for performance optimization."""
        index_definitions = [
            "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id)",
            "CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON order_items(seller_id)",
            "CREATE INDEX IF NOT EXISTS idx_orders_purchase_timestamp ON orders(order_purchase_timestamp)"
        ]
        
        for index_sql in index_definitions:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text(index_sql))
                    conn.commit()
                logger.info(f"Successfully created index: {index_sql}")
            except Exception as e:
                logger.error(f"Error creating index: {str(e)}")

def main():
    """Main execution function."""
    try:
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        
        # Initialize pipeline and analytics
        etl = ETLPipeline(engine)
        analytics = OlistAnalytics(engine)
        
        # Load data
        data_files = {
            'customers': 'customers_dataset.csv',
            'orders': 'orders_dataset.csv',
            'order_items': 'order_items_dataset.csv',
            'products': 'products_dataset.csv',
            'sellers': 'sellers_dataset.csv',
            'order_reviews': 'order_reviews_dataset.csv',
            'order_payments': 'order_payments_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }
        
        for table, file in data_files.items():
            etl.load_csv_to_db(os.path.join('data', file), table)
            
        # Create indexes for performance
        etl.create_indexes()
        
        # Generate analytics and visualizations
        analytics.create_visualizations()
        
        # Print summary metrics
        time_metrics = analytics.get_time_based_metrics()
        print("\nTime-based Metrics Summary:")
        print(time_metrics.tail().to_string())
        
        category_metrics = analytics.get_product_category_analysis()
        print("\nTop 10 Categories by Revenue:")
        print(category_metrics.head(10)[['category', 'revenue', 'avg_rating']].to_string())
        
        seller_metrics = analytics.get_seller_performance()
        print("\nSeller Performance Summary:")
        print(seller_metrics.describe().to_string())
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()