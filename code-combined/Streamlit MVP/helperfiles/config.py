# config.py

__all__ = ["STOCK_OPTIONS", "OPTIMIZATION_DISCLAIMER"]

# Mapping for stock universe options
STOCK_OPTIONS = {
    'Full S&P500 Universe': 0,
    'Top 300 S&P500 Stocks': 300,   
}

# Optimization objective explanation
OPTIMIZATION_DISCLAIMER = """
---
**Optimization Objective:**  
This portfolio is optimized to maximize the **Sharpe ratio**, balancing return and risk.
"""
