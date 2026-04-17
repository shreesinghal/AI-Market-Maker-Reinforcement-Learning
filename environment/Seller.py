import numpy as np


class Seller:
    """
    A seller that arrives at the market and decides
    whether to sell at the market maker's bid price.

    The closer the bid price is to mid-price, the more likely
    the seller is to trade.
    """

    def wants_to_trade(self, bid_price, mid_price, tick_size, rng=None):
        """
        Decide whether this seller will trade at the given bid price.

        Parameters:
            bid_price  : float, the market maker's bid (buy) price
            mid_price  : float, current mid-price of the stock
            tick_size  : float, minimum price increment
            rng        : numpy random generator, for reproducibility

        Returns:
            bool, True if the seller trades
        """
        rng = rng or np.random.default_rng()

        # How many ticks below mid is the bid?
        ticks_away = (mid_price - bid_price) / tick_size

        # Fill probability decreases as the bid moves further from mid
        # At 1 tick away: ~80% chance, decaying by 15% per additional tick
        fill_prob = max(0.0, 0.95 - 0.15 * ticks_away)

        return rng.random() < fill_prob