# FHNW - Institut für Geomatik: Masterthesis
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.02.2021, random_rect.py
# Create and plot random regular cubes


from trockenmauer.generate_stones import generate_regular_stone
from trockenmauer.plot import plot_stones

# stones = [generate_regular_stone(0.2, 0.1, 0.1, name=str(i)) for i in range(10)]
stones = [generate_regular_stone(0.2, 0.1, 0.1) for i in range(10)]

plot_stones(stones)
