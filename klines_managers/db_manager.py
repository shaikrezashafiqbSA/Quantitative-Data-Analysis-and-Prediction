import time
import pandas as pd
import datetime 
import json
import psycopg2
from psycopg2._psycopg import connection
# from utils.read_config import get_config
# from utils.secret_manager import get_db_pwd
def create_connection(db_name = "oms",
                      db_host = "oms-dev.cohexy8tb7b7.ap-southeast-1.rds.amazonaws.com",
                      db_user = "oms",
                      db_password = "TV4nBwDSc3B0wdQ9jd1x",
                      db_port = "5432") -> connection:
    
    return psycopg2.connect(database=db_name,
                            host=db_host,
                            user=db_user,
                            password=db_password,
                            port=db_port
                            )


def create_connection_fx_price_snapshots(db_name = "dtf_staging",
                                         db_host = "oms-dev.cohexy8tb7b7.ap-southeast-1.rds.amazonaws.com",
                                         db_user = "dtf_staging_user",
                                         db_password = "jw8s0F4",
                                         db_port = "5432") -> connection:
    
    """
    DB_HOST=oms-dev.cohexy8tb7b7.ap-southeast-1.rds.amazonaws.com
    DB_SCHEMA=dtf
    DB_USER=dtf_staging_user
    DB_NAME=dtf_staging
    DB_PWD=jw8s0F4
    """
    
    return psycopg2.connect(database=db_name,
                            host=db_host,
                            user=db_user,
                            password=db_password,
                            port=db_port
                            )

def fx_price_snapshot(currency="USD/SGD"):
    sql = f"SELECT * from dtf.fx_price_snapshots"
    
    conn = create_connection_fx_price_snapshots()
    cursor = conn.cursor()
    cursor.execute(sql)
    
    table = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    df = pd.DataFrame(table)
    df.columns = ["fx_pair", "bid","ask", "mid", "timestamp"]
    price=df[df["fx_pair"]==currency].iloc[-1]        
    
    return price


transactions_table_columns = ['order_id', 'cl_order_id','execution_id',\
                        'account', 'model_name','model_trade_id', \
                        'symbol', 'reporting_currency','instrument_type',\
                        'order_type','position', 'side', 'quantity', 'average_price', 'expected_price', \
                        'created_at', 'updated_at']
order_execution_table_columns = ['order_id', 'cl_order_id',	'execution_id',\
                                 'quantity', 'side', 'symbol', 'execution_status', 'order_status', 'last_price', 'average_price', \
                                 'execution_message', 'transaction_time', 'created_at', 'updated_at']
    
metrics_table_columns = ['open_at', 'close_at', 'model_name', 'account', 'symbol',\
                         'instrument_type', 'reporting_currency', 'position', 'model_trade_id', \
                         'positions', 'actual_entry_price', 'expected_entry_price', 'cost',\
                         'actual_qty', 'expected_qty', 'actual_exit_price',\
                         'expected_exit_price', 'actual_fees', 'expected_fees', 'actual_pnl_pct', \
                         'expected_pnl_pct', 'actual_rpnl', 'expected_rpnl']
    
fx_price_historical_columns = ["fx_pair", "exchange", "mid", "ts"]
    
def write_to_table(metrics: list, table = "transactions"):

    conn = create_connection()
    
    if table == "transactions":
        table_columns = transactions_table_columns
    elif table == "metrics":
        table_columns = metrics_table_columns
    
        
    sql_cols = ", ".join(["%s"]*len(table_columns))
    sql = f"INSERT INTO signal_trading.{table}" + \
          "("+", ".join(table_columns) + ")" + \
          f"VALUES ({sql_cols}); "
    cursor = conn.cursor()
    
    for metric in metrics:
        try:
            data = tuple([i for i in metric])# (metric[0], metric[1], metric[2], metric[3], metric[4], metric[5], metric[6], metric[7])
            cursor.execute(sql, data)
        except (Exception, psycopg2.Error) as error:
            print("ERROR")
            print(error)
    conn.commit()
    
    cursor.close()
    conn.close()
    pass


def read_table(table_name = "transactions"):
    assert table_name in ["transactions", "order_executions", "metrics"]
    sql = f"SELECT * from signal_trading.{table_name}"
    
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    
    table = cursor.fetchall()
    
    if table_name == "transactions":
        df = pd.DataFrame(table)
        try:
            df.columns = transactions_table_columns
        except Exception as e:
            print(f"{table_name} len: {len(df)}")
            
        if len(df) == 0:
            df = pd.DataFrame(columns = transactions_table_columns)
            return df
        for col in ['cl_order_id', 'execution_id', 'quantity', 'average_price', 'expected_price']:
            df[col] = df[col].astype(float)
            
        df["model_trade_id"] = df["model_trade_id"].astype(float)
            
    elif table_name == "order_executions":
        df = pd.DataFrame(table)
        try:
            df.columns = order_execution_table_columns
        except Exception as e:
            print(f"{table_name} len: {len(df)}")
            
            
    elif table_name == "metrics":
        df = pd.DataFrame(table)
        try:
            df.columns = metrics_table_columns
        except Exception as e:
            print(f"{table_name} len: {len(df)}")
            
    cursor.close()
    conn.close()
    return df

    
def delete_rows(table_name = "transactions", where = "model_name='TFEXUSD_5m_mock'"):
    assert table_name in ["transactions", "metrics"]
    if where is None:
        sql = f"DELETE from signal_trading.{table_name}"
    else:
        sql = f"DELETE from signal_trading.{table_name} WHERE {where}"
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    
    conn.commit()
    cursor.close()
    conn.close()

# def delete_last_rows(table_name = "transactions", n=1):
#     assert table_name in ["transactions"]
#     sql = f"DELETE from signal_trading.{table_name}" + \
#           " ORDER BY cl_order_id DESC LIMIT " + f"{n}"
    
#     conn = create_connection()
#     cursor = conn.cursor()
#     cursor.execute(sql)
    
#     conn.commit()
#     cursor.close()
#     conn.close()

def create_table(sql = None, columns=["open_at", "close_at"]):
    if sql is None:
        
        columns_sql = " VARCHAR NULL,".join(columns) + " VARCHAR NULL"
        
        sql = "CREATE TABLE IF NOT EXISTS signal_trading.metrics ( " + \
                columns_sql + \
             	");"
    conn = create_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
    except Exception as e:
        print(f"Unable to create table: {e}")

    conn.commit() 
    conn.close()
    cursor.close()
    
    
    
    
#%%
if __name__ == "__main__":
    #%%

    columns = ["fx_pair","exchange", "mid", "ts"]
        
    columns_sql = " VARCHAR NULL,".join(columns) + " VARCHAR NULL"
    
    sql = "CREATE TABLE IF NOT EXISTS dtf.fx_price_historical ( " + \
            columns_sql + \
         	");"
    conn = create_connection_fx_price_snapshots()
    cursor = conn.cursor()

    try:
        cursor.execute(sql)
    except Exception as e:
        print(f"Unable to create table: {e}")

    conn.commit() 
    conn.close()
    cursor.close()
    
