"""
Auction mechanism for ad ranking
Implements different auction mechanisms: GSP, VCG, First-Price
"""
import numpy as np
import pandas as pd


class AuctionMechanism:
    """
    Base class for auction mechanisms
    """

    def __init__(self, config):
        self.config = config
        self.mechanism = config.get('mechanism', 'gsp')
        self.reserve_price = config.get('reserve_price', 0.01)
        self.quality_weight = config.get('quality_weight', 0.5)

    def compute_rank_score(self, bids, quality_scores):
        """
        Compute rank score combining bid and quality
        rank_score = bid * quality_score^quality_weight
        """
        return bids * np.power(quality_scores, self.quality_weight)

    def run_auction(self, bids, quality_scores, ad_ids=None):
        """
        Run auction and return ranked ads with prices
        """
        if self.mechanism == 'gsp':
            return self.generalized_second_price(bids, quality_scores, ad_ids)
        elif self.mechanism == 'vcg':
            return self.vickrey_clarke_groves(bids, quality_scores, ad_ids)
        elif self.mechanism == 'first_price':
            return self.first_price_auction(bids, quality_scores, ad_ids)
        else:
            raise ValueError(f"Unknown auction mechanism: {self.mechanism}")

    def generalized_second_price(self, bids, quality_scores, ad_ids=None):
        """
        Generalized Second Price (GSP) auction
        - Widely used in search advertising (Google AdWords)
        - Each advertiser pays the minimum bid needed to maintain their position
        """
        n = len(bids)

        if ad_ids is None:
            ad_ids = np.arange(n)

        # Compute rank scores
        rank_scores = self.compute_rank_score(bids, quality_scores)

        # Sort by rank score (descending)
        sorted_indices = np.argsort(-rank_scores)

        sorted_bids = bids[sorted_indices]
        sorted_quality = quality_scores[sorted_indices]
        sorted_ad_ids = ad_ids[sorted_indices]
        sorted_rank_scores = rank_scores[sorted_indices]

        # Compute prices (GSP pricing)
        prices = np.zeros(n)

        for i in range(n):
            if i < n - 1:
                # Price is the minimum bid needed to maintain position i
                # rank_score[i] >= rank_score[i+1]
                # bid[i] * quality[i]^w >= bid[i+1] * quality[i+1]^w
                # bid[i] >= (bid[i+1] * quality[i+1]^w) / quality[i]^w
                next_rank_score = sorted_rank_scores[i + 1]
                current_quality_factor = np.power(sorted_quality[i], self.quality_weight)
                prices[i] = next_rank_score / current_quality_factor if current_quality_factor > 0 else self.reserve_price
            else:
                # Last position pays reserve price
                prices[i] = self.reserve_price

            # Apply reserve price
            prices[i] = max(prices[i], self.reserve_price)

        # Filter out ads below reserve price
        valid_mask = sorted_bids >= self.reserve_price
        filtered_ad_ids = sorted_ad_ids[valid_mask]
        filtered_prices = prices[valid_mask]
        filtered_bids = sorted_bids[valid_mask]
        filtered_quality = sorted_quality[valid_mask]
        filtered_rank_scores = sorted_rank_scores[valid_mask]

        return {
            'ad_ids': filtered_ad_ids,
            'bids': filtered_bids,
            'prices': filtered_prices,
            'quality_scores': filtered_quality,
            'rank_scores': filtered_rank_scores
        }

    def vickrey_clarke_groves(self, bids, quality_scores, ad_ids=None):
        """
        Vickrey-Clarke-Groves (VCG) auction
        - Truthful mechanism (incentive compatible)
        - Each advertiser pays their externality (social cost they impose)
        """
        n = len(bids)

        if ad_ids is None:
            ad_ids = np.arange(n)

        # Compute rank scores
        rank_scores = self.compute_rank_score(bids, quality_scores)

        # Sort by rank score
        sorted_indices = np.argsort(-rank_scores)

        sorted_bids = bids[sorted_indices]
        sorted_quality = quality_scores[sorted_indices]
        sorted_ad_ids = ad_ids[sorted_indices]
        sorted_rank_scores = rank_scores[sorted_indices]

        # Compute VCG prices
        prices = np.zeros(n)

        for i in range(n):
            # Compute total value with advertiser i
            total_value_with = np.sum(sorted_rank_scores[:n])

            # Compute total value without advertiser i
            # Remove i and re-rank
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            other_rank_scores = sorted_rank_scores[mask]

            total_value_without = np.sum(other_rank_scores)

            # Price is the externality
            prices[i] = max(total_value_without - (total_value_with - sorted_rank_scores[i]), self.reserve_price)

        # Filter by reserve price
        valid_mask = sorted_bids >= self.reserve_price
        filtered_ad_ids = sorted_ad_ids[valid_mask]
        filtered_prices = prices[valid_mask]
        filtered_bids = sorted_bids[valid_mask]
        filtered_quality = sorted_quality[valid_mask]
        filtered_rank_scores = sorted_rank_scores[valid_mask]

        return {
            'ad_ids': filtered_ad_ids,
            'bids': filtered_bids,
            'prices': filtered_prices,
            'quality_scores': filtered_quality,
            'rank_scores': filtered_rank_scores
        }

    def first_price_auction(self, bids, quality_scores, ad_ids=None):
        """
        First Price auction
        - Each advertiser pays their own bid
        - Simpler but not truthful
        """
        n = len(bids)

        if ad_ids is None:
            ad_ids = np.arange(n)

        # Compute rank scores
        rank_scores = self.compute_rank_score(bids, quality_scores)

        # Sort by rank score
        sorted_indices = np.argsort(-rank_scores)

        sorted_bids = bids[sorted_indices]
        sorted_quality = quality_scores[sorted_indices]
        sorted_ad_ids = ad_ids[sorted_indices]
        sorted_rank_scores = rank_scores[sorted_indices]

        # In first-price auction, price equals bid
        prices = sorted_bids.copy()

        # Filter by reserve price
        valid_mask = sorted_bids >= self.reserve_price
        filtered_ad_ids = sorted_ad_ids[valid_mask]
        filtered_prices = prices[valid_mask]
        filtered_bids = sorted_bids[valid_mask]
        filtered_quality = sorted_quality[valid_mask]
        filtered_rank_scores = sorted_rank_scores[valid_mask]

        return {
            'ad_ids': filtered_ad_ids,
            'bids': filtered_bids,
            'prices': filtered_prices,
            'quality_scores': filtered_quality,
            'rank_scores': filtered_rank_scores
        }

    def compute_revenue(self, auction_result):
        """Compute total revenue from auction"""
        return np.sum(auction_result['prices'])

    def compute_welfare(self, auction_result):
        """Compute social welfare (sum of values)"""
        return np.sum(auction_result['rank_scores'])


class BidGenerator:
    """
    Generate bids for ads (simulated)
    In practice, bids come from advertisers
    """

    def __init__(self, min_bid=0.01, max_bid=5.0):
        self.min_bid = min_bid
        self.max_bid = max_bid

    def generate_bids(self, n, distribution='uniform'):
        """Generate random bids"""
        if distribution == 'uniform':
            return np.random.uniform(self.min_bid, self.max_bid, n)
        elif distribution == 'exponential':
            # Exponential distribution (more realistic)
            bids = np.random.exponential(1.0, n)
            # Normalize to [min_bid, max_bid]
            bids = self.min_bid + (bids / np.max(bids)) * (self.max_bid - self.min_bid)
            return bids
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def generate_bids_from_ctr(self, predicted_ctrs, conversion_value=1.0):
        """
        Generate bids based on predicted CTR
        bid = CTR * conversion_value (simplified)
        """
        return predicted_ctrs * conversion_value
