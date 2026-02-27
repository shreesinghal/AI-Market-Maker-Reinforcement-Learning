from Stock import Stock
from Market_Maker import MarketMaker
from Buyer import Buyer
from Seller import Seller

Nvidia_Stock = Stock("NVDA", 300)
Costco_Stock = Stock("COST", 500)
Walgreen_Stock = Stock("WBA", 50)

maker_inventory = {
    Nvidia_Stock: 100,
    Costco_Stock: 100,
}

seller_inventory = {
    Walgreen_Stock: 50,
}

market_maker = MarketMaker(maker_inventory, 1000)
buyer1 = Buyer("Alice", {}, 1000)
seller1 = Seller("Bob", seller_inventory, 0)

print(f"Market Maker Cash Balance: {market_maker.cash_balance}")
print(f"Market Maker Insventory: {market_maker.inventory}")  
print(f"Buyer1 Cash Balance: {buyer1.cash_balance}")
print(f"Buyer1 Inventory: {buyer1.inventory}")
print(f"Seller1 Cash Balance: {seller1.cash_balance}")
print(f"Seller1 Inventory: {seller1.inventory}")

buyer1.attempt_buy(Nvidia_Stock, 10, market_maker)
seller1.attempt_sell(Walgreen_Stock, 5, market_maker)

print(f"Market Maker Cash Balance: {market_maker.cash_balance}")
print(f"Market Maker Inventory: {market_maker.inventory}")  
print(f"Buyer1 Cash Balance: {buyer1.cash_balance}")
print(f"Buyer1 Inventory: {buyer1.inventory}")
print(f"Seller1 Cash Balance: {seller1.cash_balance}")
print(f"Seller1 Inventory: {seller1.inventory}")