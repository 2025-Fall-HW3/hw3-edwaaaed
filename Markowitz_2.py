"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    Strategy: Momentum Sector Rotation with Inverse Volatility Weighting
    - Selects top N sectors based on momentum (cumulative returns)
    - Weights selected sectors inversely proportional to their volatility
    """
    def __init__(self, price, exclude, lookback=60, top_n=3):
        """
        Parameters:
        - lookback: Window for momentum calculation (60 days ≈ 1 quarter)
        - top_n: Number of top-performing sectors to hold (default: 3)
        """
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback 
        self.top_n = top_n

    def calculate_weights(self):
        # Get asset list (exclude SPY)
        assets = self.price.columns[self.price.columns != self.exclude]
        
        # Initialize weights DataFrame with zeros
        self.portfolio_weights = pd.DataFrame(0.0, index=self.price.index, columns=self.price.columns)

        # Pre-calculate metrics (vectorized for efficiency)
        # Momentum: Cumulative return over lookback period
        momentum = self.price[assets].pct_change(self.lookback).fillna(0)
        
        # Volatility: Standard deviation of daily returns over 20-day rolling window
        daily_rets = self.price[assets].pct_change().fillna(0)
        volatility = daily_rets.rolling(20).std().fillna(0)
        
        # Calculate weights daily (start from lookback to ensure sufficient data)
        for i in range(self.lookback, len(self.price)):
            current_date = self.price.index[i]
            
            # Get current momentum and volatility
            curr_mom = momentum.loc[current_date]
            curr_vol = volatility.loc[current_date]
            
            # === CORE STRATEGY: Select Top Performers ===
            # Choose top N sectors with highest momentum
            top_assets = curr_mom.nlargest(self.top_n).index
            
            # === RISK CONTROL: Inverse Volatility Weighting ===
            # Weight = (1/volatility) / sum(1/volatility)
            # Lower volatility → Higher weight
            
            sel_vol = curr_vol[top_assets]
            inverse_vol = 1.0 / (sel_vol + 1e-8)  # epsilon prevents division by zero
            
            # Normalize weights (ensure sum = 1)
            if inverse_vol.sum() > 0:
                weights = inverse_vol / inverse_vol.sum()
            else:
                # Fallback to equal weights if volatility is abnormal
                weights = pd.Series(1.0 / self.top_n, index=top_assets)
            
            # Assign weights (unselected assets remain 0)
            self.portfolio_weights.loc[current_date, top_assets] = weights

        # Handle missing values (first lookback days remain 0)
        self.portfolio_weights.fillna(0.0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
