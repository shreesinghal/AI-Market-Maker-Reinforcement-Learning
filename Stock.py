# Represents a Stock object in the market that has a constantly changing price.
import random 

class Stock:
    def __init__(self, abbr, price):
        self.abbr = abbr
        self.price = price
    
    def __str__(self):
        return f"{self.abbr}: ${self.price:.2f}"
    
    def __repr__(self):
        return f"Stock({self.abbr}, {self.price:.2f})"

    def get_price(self):
        # some random variation in price to simulate market changes
        variation = random.uniform(-0.05, 0.05)  # -5% to +5% variation
        return self.price * (1 + variation)
    
    def update_price(self, new_price):
        self.price = new_price