

import datetime
import json
import requests
import time
from collections import defaultdict
import pickle

# Generates a list of date time needed from Aug 8, 2013 to Jan 17, 2019 (need extra for alg)
start = datetime.datetime.strptime("04-08-2013", "%d-%m-%Y")
end = datetime.datetime.strptime("23-01-2019", "%d-%m-%Y")
date_generated = [
    start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

post_dict = defaultdict(list)
# Loops through each day
for i, date, in enumerate(date_generated):
    print("Working {}".format(date))
    # time.sleep(1)
    if i == len(date_generated) - 1:
        continue

    # Builds link
    start_date = str(int(date_generated[i].timestamp()))
    end_date = str(int(date_generated[i + 1].timestamp()))
    link = 'https://api.pushshift.io/reddit/submission/search/?after=' + start_date + \
        '&before=' + end_date + '&sort_type=score&sort=desc&subreddit=ripple'

    # Request data from api
    r = requests.get(link)

    # Pulls data from link into json object
    data = json.loads(r.text)
    # Pulls titles from each submission for that days
    for post in data['data']:
        post_dict[date].append(post['title'])


# Saves the dictionary as a pickle
with open('ripple_reddit_posts.pk', 'wb') as handle:
    pickle.dump(post_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
