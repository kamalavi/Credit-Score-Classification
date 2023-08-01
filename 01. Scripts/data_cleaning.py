import pandas as pd
import numpy as np
import regex as re
import random
import scipy
from matplotlib import pyplot as plt

#specify path for data
path = r"00. Data\\train.csv"

def data_cleaning(path):
    #import credit sore classification dataset
    df = pd.read_csv(path, low_memory=False)

    #analyze dataset
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.describe().T)

    #basic data prep
    df.columns = df.columns.str.lower()
    df = df.set_index('customer_id')

    #specify non-relevant columns/variables to drop
    df.drop(columns=['id','name','ssn','month','type_of_loan','changed_credit_limit','amount_invested_monthly'],inplace=True)

    #create helper functions to clean columns we wish to be numeric. Take series (column) as input
    def clean_num(series):
        series = series.copy()

        if series.dtype == "object":
            series = series.apply(lambda x: re.sub(r"[^0-9.]","",str(x)))
            series.replace("",np.nan,inplace=True)
            series = series.astype(float)
            
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_val = 1.5*IQR
        
        lower_bound = series.median() - outlier_val
        upper_bound = series.median() + outlier_val
        series[(series<lower_bound) | (series<0) | (series>upper_bound)] = np.nan
        

        series = series.fillna(series.groupby('customer_id').median())
        series.fillna(series.median(),inplace=True)
        
        return series

    #create helper functions to clean columns we wish to be numeric. Take series (column) as input
    def clean_string(series):
        df = series.copy().to_frame()
        df[series.name] = df[series.name].apply(lambda x: re.sub(r'[^a-zA-Z]+',"",str(x).strip()))
        df[series.name].replace("",np.nan,inplace=True) 
        grouped = df.groupby('customer_id')[series.name].agg(pd.Series.mode)
        grouped = grouped.apply(lambda x: x[round(random.uniform(0,1))] if type(x) == np.ndarray and len(x)>1 else x)
        df[series.name].fillna(grouped,inplace=True)
        global_mode = df[series.name].mode()[0]
        df[series.name] = df[series.name].apply(lambda x: global_mode if type(x) == np.ndarray and len(x)==0 else x)
        df = df.squeeze()

        return df

    #manually fix credit_history_age column
    df['credit_history_age'] = df['credit_history_age'].fillna('0 Years and 0 Months')
    df['credit_history_age_months'] = df['credit_history_age'].apply(lambda x: np.dot([int(i) for i  in re.findall("[0-9]+",x)],[12,1]))
    df['credit_history_age_months'].replace(0,np.nan,inplace=True)
    df.drop(columns='credit_history_age',inplace=True)

    #define variables to be numeric and string
    col_float = ['age',
                'annual_income',
                'monthly_inhand_salary',
                'num_bank_accounts',
                'num_credit_card',
                'interest_rate',
                'num_of_loan',
                'delay_from_due_date',
                'credit_utilization_ratio',
                'total_emi_per_month',
                'monthly_balance',
                'num_credit_inquiries',
                'num_of_delayed_payment',
                'outstanding_debt',
                'credit_history_age_months',]  

    col_string = ['credit_mix',
                'occupation',
                'payment_behaviour',
                'payment_of_min_amount']

    #clean columns
    for i in df.columns:

        if i in col_float:
            df[i] = clean_num(df[i])

        elif i in col_string:
            df[i] = clean_string(df[i])

    df.drop_duplicates().reset_index().drop(columns='customer_id',inplace=True)

    #reanalyze data to confirm desired cleaning/output (e.g., no nulls, no outstanding outliers, etc.)
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.describe().T)


    return df, col_float, col_string
