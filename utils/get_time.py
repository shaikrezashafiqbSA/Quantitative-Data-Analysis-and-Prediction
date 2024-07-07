import pytz
from datetime import datetime

def get_dt_times_now():
    dt_UTC_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    dt_SGT_now = dt_UTC_now.astimezone(pytz.timezone('Asia/Singapore'))

    microsecs = int(dt_UTC_now.microsecond / 1000)
    dt_UTC_now_reformatted = dt_UTC_now.strftime("%Y-%m-%d %H:%M:%S") + f".{microsecs:03d} UTC"

    microsecs = int(dt_SGT_now.microsecond / 1000)
    dt_SGT_now_reformatted = dt_SGT_now.strftime("%Y-%m-%d %H:%M:%S") + f".{microsecs:03d} SGT"

    return dt_UTC_now_reformatted, dt_SGT_now_reformatted