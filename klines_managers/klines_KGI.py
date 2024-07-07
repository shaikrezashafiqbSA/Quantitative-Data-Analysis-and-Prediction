import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2._psycopg import connection
from settings import DB_USER, DB_NAME, DB_PASSWORD, DB_HOST,DB_PORT, DB_SCHEMA 

def create_connection(db_name = DB_NAME,
                      db_host = DB_HOST,
                      db_user = DB_USER,
                      db_password = DB_PASSWORD,
                      db_port = DB_PORT) -> connection:
    
    return psycopg2.connect(database=db_name,
                            host=db_host,
                            user=db_user,
                            password=db_password,
                            port=db_port
                            )


def query_table(conn, 
                table_name = 'fx_aggregate_1m', 
                instrument= "USD/SGD", 
                schema_name = DB_SCHEMA,
                print_table_columns= False,
                since=None,
                to=None
                ):
    # create a new cursor
    with conn.cursor() as cur:
        # write the SQL query to get the column names
        if print_table_columns:
            column_query = "SELECT column_name FROM information_schema.columns WHERE table_name = '{}'".format(table_name)
            # execute the SQL query
            cur.execute(column_query)
            # fetch all the column names
            columns = cur.fetchall()
            # print the column names
            for column in columns:
                print(column[0])

        # write the SQL query to get the data
        select_query = "SELECT * FROM {}.{} WHERE name = '{}'".format(schema_name, table_name, instrument)
        if since and to:
            select_query += " AND ts BETWEEN '{}' AND '{}'".format(since, to)
        # execute the SQL query
        cur.execute(select_query)
        # fetch all the rows
        rows = cur.fetchall()
        
    return rows




def load_ohlcv(table_name = 'fx_aggregate_1m', 
                instrument= "USD/SGD", 
                schema_name = DB_SCHEMA,
                print_table_columns= False,
                since=None,
                to=None
                ):
    # create a connection
    conn = create_connection()
    # query the table
    rows = query_table(conn, 
                       table_name = table_name,
                        instrument= instrument, 
                        schema_name = schema_name,
                        print_table_columns= print_table_columns,
                        since=since,
                        to=to)
    # close the connection
    conn.close()

    # convert the data into pandas dataframe
    columns = ["Currency", "Datetime", "open", "high", "low", "close"]
    df = pd.DataFrame(rows[1:], columns=columns)
    df=df[columns[1:]]
    df.set_index('Datetime', inplace=True)
    # Convert decimal type open, high, low, close to float
    for col in ["open", "high","low","close"]:
        df[col] = df[col].astype(float)
    # Check for any zeros and replace them with nan
    df = df.replace(0, float("NaN"))
    # make column of zeros for volume
    df["volume"] = 0
    df["close_time"] = df.index.astype('int64')/1e6
    df.index = df.index.tz_localize(None)
    return df


def create_connection_fx_price_snapshots(db_name = DB_NAME,
                                         db_host = DB_HOST,
                                         db_user = DB_USER,
                                         db_password = DB_PASSWORD,
                                         db_port = DB_PORT) -> connection:
    
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
    sql = f"SELECT * from {DB_SCHEMA}.fx_price_snapshots"
    
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


def query_fx_price_snapshots(conn, currency="USD/SGD"):
    sql = f"SELECT * from {DB_SCHEMA}.fx_price_snapshots"
    
    with conn.cursor() as cur:
        cur.execute(sql)
        table = cur.fetchall()
    
    df = pd.DataFrame(table)
    df.columns = ["fx_pair", "bid","ask", "mid", "timestamp"]
    price = df[df["fx_pair"] == currency].iloc[-1]        
    
    return price


def load_fx_price_snapshots(currency="USD/SGD"):
    conn = create_connection_fx_price_snapshots()
    price = query_fx_price_snapshots(conn, currency)
    conn.close()
    
    return price
