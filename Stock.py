import random# Represents a Stock object in the market that has a constantly changing price.

class Stock:
    def __init__(self, abbr, price, volatility=0.01, drift=0.0):
        """
        abbr : String, ticker symbol for the stock (e.g. 'AAPL').
        price : float, Initial mid-price of the stock.
        volatility : float, For example, 0.01 ≈ 1% volatility per step.
        drift : float, 
            Expected average return per step.  It is 0 for a pure random walk,
            but we can set it to a positive value to simulate a trend.
            For example, 0.01 ≈ 1% drift per step.
        """
        self.abbr = abbr
        self.price = float(price)
        self.volatility = float(volatility)
        self.drift = float(drift)
    
    def __str__(self):
        return f"{self.abbr}: ${self.price:.2f}"
    
    def __repr__(self):
        return f"Stock({self.abbr}, {self.price:.2f})"

    def get_price(self):
        return self.price

    # the price changes over time,
    # when we want to advance the true price in our environment 
    #we can call self.stock.step() 
    def step(self, rng):
        """
        Advance the price one step using a Gaussian random walk.
        rng : numpy random Generator (e.g. env.np_random)
        """
        shock = float(rng.normal(self.drift, self.volatility))
        self.price *= (1 + shock)

        # Prevent negative price
        if self.price <= 0:
            self.price = 0.01