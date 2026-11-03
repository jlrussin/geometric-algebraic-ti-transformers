"""Transformer experiments for "Geometric and algebraic inference in humans and neural networks".

One canonical pretraining configuration, one finetuning configuration, and one
two-factor test battery; every experiment in the paper is a named delta on them.
"""

__version__ = "0.1.0"
# Accuracy at chance: a two-way choice answered by a coin.
CHANCE_ACCURACY: float = 0.5
