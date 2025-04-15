# view/__init__.py

from .view_top300 import render as render_top300
from .view_full import render as render_full
from .view_sentiment import render as render_sentiment
from .view_backtest import render as render_backtest

__all__ = ['render_top300', 'render_full', 'render_sentiment', 'render_backtest']
