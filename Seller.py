class Seller:
    def __init__(self, name):
        self.name = name
        self.inventory = {}

    def add_stock(self, stock, quantity):
        if stock in self.inventory:
            self.inventory[stock] += quantity
        else:
            self.inventory[stock] = quantity

    def remove_stock(self, stock, quantity):
        if stock in self.inventory and self.inventory[stock] >= quantity:
            self.inventory[stock] -= quantity
            return True
        return False