# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.02.2021, random_rect.py
# Create and plot random regular cubes


from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import plot_stones

"""
Generates regular cuboids with a length within ``[0.05, 0.3]``
and scales them randomly with a factor within ``[0.5, 2]``.
This results in a minimal length of 0.05 and a maximal length of 0.6.
"""

# stones = [generate_regular_stone(0.2, 0.1, 0.1, name=str(i)) for i in range(10)]
stones = [generate_regular_stone(0.2, 0.1, 0.1, edge_noise=0.5, scale=[1, 2]) for i in range(10)]

plot_stones(stones)
