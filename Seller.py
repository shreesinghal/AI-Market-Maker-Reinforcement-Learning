import numpy as np


class Seller:
    """
    Simple seller model.
    Fill probability falls as bid moves away from mid.
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

        # Quote distance from mid in ticks.
        ticks_away = (mid_price - bid_price) / tick_size

        # Smoothly decreasing fill curve.
        fill_prob = 0.55 / (1.0 + np.exp((ticks_away - 0.5) / 0.9))

        return rng.random() < fill_prob