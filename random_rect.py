# FHNW - Institut für Geomatik: Masterthesis 
# Maschinelles Lernen für die digitale Konstruktion von Trockenmauern
# Stefan Hochuli, 24.02.2021
# random_rect.py
# Desc: Create and plot a random rectangle, calculate characteristics

from trockenmauer.synth_stone import generate_regular_stone


for i in range(10):
    stone = generate_regular_stone(0.2, 0.1, 0.1)
    stone.plot()