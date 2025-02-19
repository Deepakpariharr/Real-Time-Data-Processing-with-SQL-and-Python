# Import necessary libraries
import pandas as pd                 # For data manipulation and analysis
import numpy as np                  # For numerical operations
from sqlalchemy import create_engine, text  # For database operations
from dotenv import load_dotenv      # For loading environment variables
import logging                      # For logging system operations
import os                          # For operating system operations
import matplotlib.pyplot as plt     # For creating visualizations
import seaborn as sns              # For enhanced visualizations
from datetime import datetime       # For date/time operations
from typing import Dict, List, Tuple, Optional  # For type hints
import json                        # For JSON operations

# Set up logging to track what our program is doing
logging.basicConfig(
    level=logging.INFO,  # Show all info level messages
    format='%(asctime)s - %(levelname)s - %(message)s'  # Include timestamp and message type
)
logger = logging.getLogger(__name__)

class DataQualityChecker:
    """
    This class handles all data quality checks to ensure our data is reliable.
    Think of it as a quality control inspector for our data.
    """
    
    def __init__(self, engine):
        """
        Initialize with a database connection.
        Parameters:
            engine: SQLAlchemy engine for database connection
        """
        self.engine = engine
        
    def check_nulls(self, table_name: str) -> Dict[str, float]:
        """
        Checks what percentage of each column contains null values.
        This helps us identify missing data problems.
        
        Parameters:
            table_name: Name of the table to check
            
        Returns:
            Dictionary with column names and their null percentages
        """
        query = f"""
            SELECT 
                column_name,
                -- Calculate percentage of nulls
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
    """
    This class handles all our business analytics.
    It's like a business analyst that answers questions about our data.
    """
    
    def __init__(self, engine):
        """
        Initialize with a database connection.
        Parameters:
            engine: SQLAlchemy engine for database connection
        """
        self.engine = engine
    
    def analyze_sales_patterns(self) -> pd.DataFrame:
        """
        Analyzes sales patterns over time, including:
        - Daily/monthly sales trends
        - Revenue patterns
        - Order volume changes
        
        Returns:
            DataFrame with sales metrics over time
        """
        query = """
            -- First, group all sales by month
            WITH monthly_sales AS (
                SELECT 
                    -- Convert timestamp to month for grouping
                    DATE_TRUNC('month', o.order_purchase_timestamp) as month,
                    -- Count total orders
                    COUNT(DISTINCT o.order_id) as order_count,
                    -- Sum up revenue
                    SUM(oi.price) as revenue,
                    -- Count unique customers
                    COUNT(DISTINCT o.customer_id) as customer_count
                FROM orders o
                JOIN order_items oi ON o.order_id = oi.order_id
                WHERE o.order_status = 'delivered'  -- Only count completed orders
                GROUP BY month
            )
            -- Then calculate additional metrics
            SELECT 
                month,
                order_count,
                revenue,
                customer_count,
                -- Calculate average order value
                revenue / NULLIF(order_count, 0) as avg_order_value,
                -- Calculate revenue per customer
                revenue / NULLIF(customer_count, 0) as revenue_per_customer
            FROM monthly_sales
            ORDER BY month
        """
        return pd.read_sql(query, self.engine)
    
    def analyze_customer_behavior(self) -> pd.DataFrame:
        """
        Analyzes customer purchasing patterns, including:
        - Purchase frequency
        - Average order value
        - Customer location analysis
        
        Returns:
            DataFrame with customer behavior metrics
        """
        query = """
            WITH customer_metrics AS (
                SELECT 
                    c.customer_id,
                    c.customer_state,
                    -- Count their orders
                    COUNT(DISTINCT o.order_id) as order_count,
                    -- Sum their total spend
                    SUM(oi.price) as total_spend,
                    -- Calculate their average order value
                    AVG(oi.price) as avg_order_value,
                    -- Find their first order date
                    MIN(o.order_purchase_timestamp) as first_order,
                    -- Find their most recent order date
                    MAX(o.order_purchase_timestamp) as last_order
                FROM customers c
                JOIN orders o ON c.customer_id = o.customer_id
                JOIN order_items oi ON o.order_id = oi.order_id
                GROUP BY c.customer_id, c.customer_state
            )
            SELECT 
                customer_state,
                -- Count customers in each state
                COUNT(*) as customer_count,
                -- Calculate average orders per customer
                AVG(order_count) as avg_orders_per_customer,
                -- Calculate average customer lifetime value
                AVG(total_spend) as avg_customer_value,
                -- Calculate average order value
                AVG(avg_order_value) as avg_order_value
            FROM customer_metrics
            GROUP BY customer_state
            ORDER BY customer_count DESC
        """
        return pd.read_sql(query, self.engine)
    
    def analyze_product_performance(self) -> pd.DataFrame:
        """
        Analyzes how different products are performing, including:
        - Sales volume
        - Revenue generation
        - Customer satisfaction (reviews)
        
        Returns:
            DataFrame with product performance metrics
        """
        query = """
            SELECT 
                -- Get English category name if available
                COALESCE(ct.product_category_name_english, 
                        p.product_category_name) as category,
                -- Count products in category
                COUNT(DISTINCT p.product_id) as product_count,
                -- Count total sales
                COUNT(DISTINCT oi.order_id) as order_count,
                -- Calculate total revenue
                SUM(oi.price) as total_revenue,
                -- Calculate average price
                AVG(oi.price) as avg_price,
                -- Get average rating
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
        """
        Creates various visualizations to help understand the data.
        Saves all plots to the specified output directory.
        
        Parameters:
            output_dir: Directory where plots should be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Sales Trends Visualization
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
        
        # 2. Customer Distribution Map
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
    """
    This class handles the Extract, Transform, Load (ETL) process.
    It's responsible for getting data from CSV files into our database.
    """
    
    def __init__(self, engine):
        """
        Initialize with a database connection.
        Parameters:
            engine: SQLAlchemy engine for database connection
        """
        self.engine = engine
        self.data_quality = DataQualityChecker(engine)
    
    def load_csv_to_db(self, file_path: str, table_name: str) -> None:
        """
        Loads a CSV file into the database and performs quality checks.
        
        Parameters:
            file_path: Path to the CSV file
            table_name: Name of the table to create in database
        """
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Load it into the database
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            logger.info(f"Successfully loaded {file_path} into {table_name}")
            
            # Perform quality checks
            null_checks = self.data_quality.check_nulls(table_name)
            if null_checks:
                logger.info(f"Null check results for {table_name}: {json.dumps(null_checks, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

def main():
    """
    Main function that runs our entire analysis pipeline.
    """
    try:
        # Load database configuration
        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        
        # Initialize our main classes
        etl = ETLPipeline(engine)
        analytics = OlistAnalytics(engine)
        
        # Define our data files
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
        
        # Load each data file
        for table, file in data_files.items():
            etl.load_csv_to_db(os.path.join('data', file), table)
        
        # Run analytics
        print("\nAnalyzing sales patterns...")
        sales_metrics = analytics.analyze_sales_patterns()
        print(sales_metrics.tail().to_string())
        
        print("\nAnalyzing customer behavior...")
        customer_metrics = analytics.analyze_customer_behavior()
        print(customer_metrics.head().to_string())
        
        print("\nAnalyzing product performance...")
        product_metrics = analytics.analyze_product_performance()
        print(product_metrics.head().to_string())
        
        # Create visualizations
        print("\nCreating visualizations...")
        analytics.create_visualizations()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()