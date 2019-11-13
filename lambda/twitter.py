from twython import Twython


class Twitter():
    def __init__(self, CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_SECRET):
        self.CONSUMER_KEY = CONSUMER_KEY
        self.CONSUMER_SECRET = CONSUMER_SECRET
        self.ACCESS_TOKEN = ACCESS_TOKEN
        self.ACCESS_SECRET = ACCESS_SECRET
        self.twitter = Twython(self.CONSUMER_KEY, self.CONSUMER_SECRET, self.ACCESS_TOKEN, self.ACCESS_SECRET)

    def update_status(self, status):
        self.twitter.update_status(status=status)
