class MarketMaker:
    def __init__(self, inventory = {}, cash_balance = 0):
        self.inventory = inventory # Dictionary of stocks and their quantities
        self.cash_balance = cash_balance

    def quote(self, stock):
        # market maker provides a bid and ask price for the stock
        bid_price = stock.get_price() * 0.9  # FOR NOW
        ask_price = stock.get_price() * 1.1   # FOR NOW
        return bid_price, ask_price
    
    def ask_price(self, stock):
        _, ask_price = self.quote(stock)
        return ask_price
    
    def bid_price(self, stock):
        bid_price, _ = self.quote(stock)
        return bid_price

    def can_sell(self, stock, amount):
        # Check if the market maker has enough of the specified stock and amount of it to sell
        return self.inventory.get(stock, 0) >= amount

    def sell_stock(self, stock, amount):
        if self.can_sell(stock, amount):
            self.inventory[stock] -= amount
            _, stock_ask_price = self.quote(stock)
            self.cash_balance += stock_ask_price * amount
            return True
        return False
    
    def can_buy(self, stock, amount):
        # Check if the market maker has enough cash to buy the specified amount of the stock
        bid_price, _ = self.quote(stock)
        return self.cash_balance >= bid_price * amount
    
    def buy_stock(self, stock, amount):
        if self.can_buy(stock, amount):
            bid_price, _ = self.quote(stock)
            self.cash_balance -= bid_price * amount
            self.inventory[stock] = self.inventory.get(stock, 0) + amount
            return True
        return False