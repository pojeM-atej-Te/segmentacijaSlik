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
    height, width, channels = slika.shape

    # Pripravi prostor značilnic
    if dimenzija == 3:  # samo barve
        znacilnice = slika.reshape(-1, 3).astype(np.float64)
    else:  # barve + lokacije
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        x_coords = x_coords.flatten() * 255.0 / width
        y_coords = y_coords.flatten() * 255.0 / height
        barve = slika.reshape(-1, 3).astype(np.float64)
        znacilnice = np.column_stack((x_coords, y_coords, barve))

    vzorcenje = 10
    vzorcene_tocke = znacilnice[::vzorcenje]
    max_iteracije = 10
    premaknjene_tocke = []

    for i, tocka in enumerate(vzorcene_tocke):
        trenutna_tocka = tocka.copy()

        for _ in range(max_iteracije):
            razdalje = np.array([manhattanska_razdalja(trenutna_tocka, t) for t in znacilnice])
            utezi = gaussovo_jedro(razdalje, velikost_okna)
            nova_tocka = np.sum(utezi[:, np.newaxis] * znacilnice, axis=0) / np.sum(utezi)

            if manhattanska_razdalja(nova_tocka, trenutna_tocka) < 0.1:
                break

            trenutna_tocka = nova_tocka

        premaknjene_tocke.append(trenutna_tocka)

    min_cd = velikost_okna / 2
    centri = []

    for tocka in premaknjene_tocke:
        dodan = False
        for i, center in enumerate(centri):
            if manhattanska_razdalja(tocka, center) < min_cd:
                centri[i] = (centri[i] + tocka) / 2
                dodan = True
                break

        if not dodan:
            centri.append(tocka)

    centri = np.array(centri)
    oznake = np.zeros(znacilnice.shape[0], dtype=int)
    for i, tocka in enumerate(znacilnice):
        najboljsi_center = 0
        najmanjsa_razdalja = float('inf')

        for j, center in enumerate(centri):
            razdalja = manhattanska_razdalja(tocka, center)
            if razdalja < najmanjsa_razdalja:
                najmanjsa_razdalja = razdalja
                najboljsi_center = j

        oznake[i] = najboljsi_center

    if dimenzija == 3:
        barve_centrov = centri
    else:
        barve_centrov = centri[:, 2:5]

    segmentirana_slika = np.zeros((height * width, 3))
    for i in range(len(centri)):
        segmentirana_slika[oznake == i] = barve_centrov[i]

    return segmentirana_slika.reshape((height, width, 3)).astype(np.uint8)

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    h, w, c = slika.shape
    k = 3  # Privzeto število centrov

    if izbira == "ročno":
        # Implementacija ročne izbire centrov
        def on_click(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                centri.append(np.array(slika[y, x]))
                print(f"Izbrana barva: {slika[y, x]}")

        centri = []
        okno = "Izberi centre"
        cv.namedWindow(okno)
        cv.setMouseCallback(okno, on_click)

        cv.imshow(okno, cv.cvtColor(slika, cv.COLOR_BGR2RGB))

        while len(centri) < k:
            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC za izhod
                break

        cv.destroyAllWindows()
        centri = np.array(centri)

    else:  # naključna izbira
        if dimenzija_centra == 3:  # samo barve
            centri = []
            while len(centri) < k:
                # Naključno izberi piksel
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                novi_center = slika[y, x].astype(np.float64)

                # Preveri razdaljo do obstoječih centrov
                if not centri:
                    centri.append(novi_center)
                else:
                    dodaj = True
                    for center in centri:
                        if manhattanska_razdalja(center, novi_center) < T:
                            dodaj = False
                            break
                    if dodaj:
                        centri.append(novi_center)

            centri = np.array(centri)

        elif dimenzija_centra == 5:  # barve + lokacije
            centri = []
            while len(centri) < k:
                # Naključno izberi piksel
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                barva = slika[y, x].astype(np.float64)

                # Normaliziraj koordinate na podoben razpon kot barve
                y_norm = y * 255.0 / h
                x_norm = x * 255.0 / w

                novi_center = np.concatenate(([x_norm, y_norm], barva))

                # Preveri razdaljo do obstoječih centrov
                if not centri:
                    centri.append(novi_center)
                else:
                    dodaj = True
                    for center in centri:
                        if manhattanska_razdalja(center, novi_center) < T:
                            dodaj = False
                            break
                    if dodaj:
                        centri.append(novi_center)

            centri = np.array(centri)

    return centri

def manhattanska_razdalja(vektor1, vektor2):
    """Izračuna manhattansko razdaljo med dvema vektorjema."""
    return np.sum(np.abs(vektor1 - vektor2))

def gaussovo_jedro(d, h):
    """Izračuna gaussovo jedro."""
    return np.exp(-d**2 / (2 * h**2))

if __name__ == "__main__":
    pass