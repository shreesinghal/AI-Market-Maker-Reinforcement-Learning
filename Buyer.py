class Buyer:
    def __init__(self, name, inventory = {}, cash_balance = 10000):
        self.name = name
        self.inventory = inventory
        self.cash_balance = cash_balance

    def calculate_acceptable_bid_price_for_stock(self, stock) -> float:
        # This function calculates the maximum price the buyer is willing to pay for a stock
        return stock.get_price() * 1.10  # FOR NOW

    def attempt_buy(self, stock, amount, market_maker) -> bool:    
        if market_maker.ask_price(stock) * amount > self.cash_balance:
            print(f"{self.name} does not have enough cash to buy {amount} units of {stock.abbr}")
            return False

        acceptable_price = self.calculate_acceptable_bid_price_for_stock(stock)
        if market_maker.ask_price(stock) > acceptable_price:
            print(f"{self.name} finds the price too high to buy {stock.abbr}")
            return False
        if market_maker.sell_stock(stock, amount):
            self.cash_balance -= market_maker.ask_price(stock) * amount
            return True
        return False