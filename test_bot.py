from apscheduler.schedulers.background import BlockingScheduler, BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import numpy as np
from utils.get_time import get_dt_times_now
import time
from datetime import datetime
class test_bot:
    def __init__(self):
        pass

    def run_bot_cycle(self):
        dt_UTC_str, dt_SGT_str = get_dt_times_now()
        # only run between 0000 to 0900 UTC
        dt_UTC = datetime.strptime(dt_UTC_str, "%Y-%m-%d %H:%M:%S.%f UTC")

        if int(dt_UTC.hour) >9:
            print(f"{dt_UTC} --> NOT PROCESSING BOT CYCLE")
            return

        print(f"{dt_UTC} --> PROCESSING BOT CYCLE")
        time.sleep(5)
        dt_UTC, dt_SGT = get_dt_times_now()
        print(f"{dt_UTC} -->  PROCESSED BOT CYCLE")

    def run_bot_update(self):  # added self here
        dt_UTC, dt_SGT = get_dt_times_now()
        print(f"{dt_UTC} ==> BOT UPDATE")

    def start_jobs(self,second_delay=1):
        # sched = AsyncIOScheduler()
        sched = BlockingScheduler()

        second1 = '*'
        minutes1 = ','.join([str(int(x)) for x in np.arange(0,60,1) if x<60])
        hours1 = "0,1,2,3,4,5,7,8,9"
    
        
        # print(f"\nrun_bot_cycle job added\nsecond: {second1}\nminutes: {minutes1}\nhours: {hours1}")

        second2 = '*'
        minutes2 = ','.join([str(int(x)) for x in np.arange(0,60,30) if x<60])
        hours2 = "0,1,2,3,4,5,7,8,9"
        
        # print(f"\nrun_bot_update job added\nsecond: {second2}\nminutes: {minutes2}\nhours: {hours2}")

        # wait for start of minute to start jobs.
        time.sleep(60+second_delay - datetime.now().second)

        dt_UTC, dt_SGT = get_dt_times_now()
        print(f"---- Starting jobs AT {dt_UTC} ----")
        sched.add_job(self.run_bot_cycle, IntervalTrigger(minutes=1))
        # sched.add_job(self.run_bot_cycle, CronTrigger(hour=hours1, minute=minutes1,))
        sched.add_job(self.run_bot_update, CronTrigger(hour=hours2, minute=minutes2, second=second2,))
        sched.start()


import schedule
import time
import requests
from datetime import datetime

# Function to send the heartbeat
def send_heartbeat(message):
    # Get the current date and time
    current_time = datetime.now()
    slack_webhook_url = 'https://hooks.slack.com/services/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'

    payload = {
        "text": f"{message} Pulse: {current_time}"
    }
    # Schedule the heartbeat job from 8 am to 5 pm (every hour)
    # if datetime.strptime('08:00:00', '%H:%M:%S').time() <= current_time <= datetime.strptime('17:00:00', '%H:%M:%S').time():
    # print(message, " - Pulse:", current_time)
    try:
        response = requests.post(slack_webhook_url, json=payload)
        if response.status_code == 200:
            print(f"Sent '{message}' heartbeat to Slack at {current_time}")
        else:
            print(f"Failed to send '{message}' heartbeat to Slack. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Failed to send '{message}' heartbeat to Slack. Exception: {e}")

# Main loop to run the scheduler
while True:
    # schedule.run_pending()
    # time.sleep(1)  # Check for scheduled jobs every 60 seconds

    # Schedule the partial functions at specific times
    schedule.every().minute.at(":01").do(send_heartbeat, message='60 Minute Check')
    schedule.every().minute.at(":02").do(send_heartbeat, message='15 Minute Check')
    schedule.every().minute.at(":03").do(send_heartbeat, message='30 Minute Check')
    schedule.every().minute.at(":04").do(send_heartbeat, message='45 Minute Check')

    # Main loop to run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)  # Check for scheduled jobs every 60 seconds
if __name__ == "__main__":
    test_bot = test_bot()
    test_bot.start_jobs()
    
