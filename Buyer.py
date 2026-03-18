import numpy as np


class Buyer:
    """
    A buyer that arrives at the market and decides
    whether to buy at the market maker's ask price.

    The closer the ask price is to mid-price, the more likely
    the buyer is to trade.
    """

    def wants_to_trade(self, ask_price, mid_price, tick_size, rng=None):
        """
        Decide whether this buyer will trade at the given ask price.

        Parameters:
            ask_price  : float, the market maker's ask (sell) price
            mid_price  : float, current mid-price of the stock
            tick_size  : float, minimum price increment
            rng        : numpy random generator, for reproducibility

        Returns:
            bool, True if the buyer trades
        """
        rng = rng or np.random.default_rng()

        # How many ticks above mid is the ask?
        ticks_away = (ask_price - mid_price) / tick_size

        # Fill probability decreases as the ask moves further from mid
        # At 1 tick away: ~80% chance, decaying by 15% per additional tick
        fill_prob = max(0.0, 0.95 - 0.15 * ticks_away)

        return rng.random() < fill_prob