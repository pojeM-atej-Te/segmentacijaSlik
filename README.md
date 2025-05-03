Ta projekt implementira algoritma K-means in Mean-Shift za segmentacijo slik.

## Funkcionalnosti
- K-means segmentacija
- Mean-Shift segmentacija
- Različne metode izbire začetnih centrov

## Namestitev
pip install -r requirements.txt

## Uporaba
```python
import cv2 as cv
from src.segmentacija import kmeans, meanshift

# Naloži sliko
slika = cv.imread("pot_do_slike.jpg")

# K-means segmentacija
segmentirana_kmeans = kmeans(slika, k=5, iteracije=10)

# Mean-Shift segmentacija
segmentirana_meanshift = meanshift(slika, velikost_okna=30, dimenzija=3)