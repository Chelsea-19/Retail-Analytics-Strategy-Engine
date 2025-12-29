import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker and set random seed
fake = Faker()
np.random.seed(42)  

# -Setting parameters
NUM_CUSTOMERS = 1000
NUM_TRANSACTIONS = 5000

# 1. Generating User Profile Data
def generate_customers(n):
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n)],
        'join_date': [fake.date_between(start_date='-3y', end_date='today') for _ in range(n)],
        'age': np.random.randint(18, 65, size=n),
        'gender': np.random.choice(['M', 'F', 'Other'], size=n, p=[0.45, 0.5, 0.05]),
        'member_tier': np.random.choice(['Bronze', 'Silver', 'Gold'], size=n, p=[0.7, 0.2, 0.1]),
        'region': np.random.choice(['Singapore', 'Malaysia', 'Hong Kong'], size=n, p=[0.7, 0.2, 0.1])
    }
    return pd.DataFrame(data)

# 2. Generating Transaction History Data
def generate_transactions(n_trans, customer_ids):
    products = ['Oolong Tea', 'Green Tea', 'Black Tea', 'Premium Pu-erh', 'Floral Blend']
    channels = ['Online Store', 'Funan Mall Outlet'] 
    
    data = {
        'transaction_id': [f'TXN_{i:08d}' for i in range(n_trans)],
        'customer_id': np.random.choice(customer_ids, size=n_trans),
        'transaction_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(n_trans)],
        'product_category': np.random.choice(products, size=n_trans),
        'quantity': np.random.randint(1, 5, size=n_trans),
        'unit_price': np.random.uniform(10, 100, size=n_trans).round(2),
        'channel': np.random.choice(channels, size=n_trans, p=[0.6, 0.4]), 
        'is_churned': np.random.choice([0, 1], size=n_trans, p=[0.9, 0.1]) 
    }
    df = pd.DataFrame(data)
    df['total_amount'] = df['quantity'] * df['unit_price']
    return df

# Generate mock data
print("Generating Mock Data...")
df_customers = generate_customers(NUM_CUSTOMERS)
df_transactions = generate_transactions(NUM_TRANSACTIONS, df_customers['customer_id'])

# Save as csv
df_customers.to_csv('C:/Users/LDD/Desktop/mock_customer_profile.csv', index=False)
df_transactions.to_csv('C:/Users/LDD/Desktop/mock_transaction_history.csv', index=False)

print("Mock Data Generated Successfully!")
print(df_customers.head())