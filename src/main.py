import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    h, w, c = slika.shape
    slika_flat = slika.reshape(-1, c).astype(np.float64)

    # Inicializacija centrov
    centri = izracunaj_centre(slika, "naključno", 3, T=30)

    # Omejimo število centrov na k
    if len(centri) > k:
        centri = centri[:k]

    # Če nimamo dovolj centrov, dodamo naključne
    while len(centri) < k:
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        novi_center = slika[y, x].astype(np.float64)
        centri = np.vstack([centri, novi_center])

    for i in range(iteracije):
        # Izračun razdalj in dodelitev klastrov
        razdalje = np.zeros((slika_flat.shape[0], k))

        for i_center, center in enumerate(centri):
            for i_piksel, piksel in enumerate(slika_flat):
                razdalje[i_piksel, i_center] = manhattanska_razdalja(piksel, center)

        # Določi klaster za vsak piksel
        oznake = np.argmin(razdalje, axis=1)

        # Posodobi centre
        novi_centri = np.zeros_like(centri)
        for i_center in range(k):
            piksli_v_klastru = slika_flat[oznake == i_center]
            if len(piksli_v_klastru) > 0:
                novi_centri[i_center] = np.mean(piksli_v_klastru, axis=0)
            else:
                novi_centri[i_center] = centri[i_center]

        # Preveri konvergenco
        if np.allclose(novi_centri, centri):
            break

        centri = novi_centri

    # Ustvari segmentirano sliko
    segmentirana_slika = np.zeros_like(slika_flat)
    for i_center in range(k):
        segmentirana_slika[oznake == i_center] = centri[i_center]

    return segmentirana_slika.reshape(slika.shape).astype(np.uint8)

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

def manhattanska_razdalja(vektor1, vektor2):
    """Izračuna manhattansko razdaljo med dvema vektorjema."""
    return np.sum(np.abs(vektor1 - vektor2))

if __name__ == "__main__":
    pass