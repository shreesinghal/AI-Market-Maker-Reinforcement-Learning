class Seller:
    def __init__(self, name, inventory = {}, cash_balance = 0):
        self.name = name
        self.inventory = inventory
        self.cash_balance = cash_balance

    def calculate_acceptable_ask_price_for_stock(self, stock) -> float:
        # This function calculates the minimum price the seller is willing to accept for a stock
        return stock.get_price() * 0.90  # FOR NOW

    def attempt_sell(self, stock, amount, market_maker) -> bool:
        if stock not in self.inventory or self.inventory[stock] < amount:
            print(f"{self.name} does not have enough {stock.abbr} to sell.")
            return False

        acceptable_price = self.calculate_acceptable_ask_price_for_stock(stock)
        if market_maker.bid_price(stock) < acceptable_price:
            print(f"{self.name} finds the price too low to sell {stock.abbr}")
            return False
        if market_maker.buy_stock(stock, amount):
            self.inventory[stock] -= amount
            self.cash_balance += market_maker.bid_price(stock) * amount
            return True
        return False