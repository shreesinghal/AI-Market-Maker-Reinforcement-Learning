# Represents a Stock object in the market that has a constantly changing price.
import random

class Stock:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def get_price(self):
        # some random variation in price to simulate market changes
        variation = random.uniform(-0.05, 0.05)  # -5% to +5% variation
        return self.price * (1 + variation)