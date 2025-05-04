import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


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

    # Vectorized k-means
    for _ in range(iteracije):
        # Izračun razdalj in dodelitev klastrov - vektorizirano
        razdalje = np.zeros((slika_flat.shape[0], k))

        for i_center in range(k):
            razdalje[:, i_center] = np.sum(np.abs(slika_flat - centri[i_center]), axis=1)

        # Določi klaster za vsak piksel
        oznake = np.argmin(razdalje, axis=1)

        # Posodobi centre
        novi_centri = np.zeros_like(centri)
        for i_center in range(k):
            maska = oznake == i_center
            if np.any(maska):
                novi_centri[i_center] = np.mean(slika_flat[maska], axis=0)
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
    '''Izvede segmentacijo slike z uporabo metode mean-shift.
    Optimizirana verzija za boljše rezultate.'''
    height, width, channels = slika.shape

    # Zmanjšamo velikost slike za hitrejše procesiranje, ampak ne preveč
    scale_factor = 0.5
    resized_img = cv.resize(slika, (int(width * scale_factor), int(height * scale_factor)))
    r_height, r_width = resized_img.shape[:2]

    # Pripravi prostor značilnic na manjši sliki
    if dimenzija == 3:  # samo barve
        znacilnice = resized_img.reshape(-1, 3).astype(np.float64)
    else:  # barve + lokacije
        x_coords, y_coords = np.meshgrid(np.arange(r_width), np.arange(r_height))
        x_coords = x_coords.flatten() * 255.0 / r_width
        y_coords = y_coords.flatten() * 255.0 / r_height
        barve = resized_img.reshape(-1, 3).astype(np.float64)
        znacilnice = np.column_stack((x_coords, y_coords, barve))

    # Zmerno vzorčenje za hitrejše procesiranje
    vzorcenje = 20  # Zmanjšano iz 50 na 20
    vzorcene_tocke = znacilnice[::vzorcenje]
    print(f"Število vzorčnih točk: {len(vzorcene_tocke)}")

    # Povečamo maksimalno število vzorčnih točk
    max_vzorcne_tocke = 300  # Povečano iz 100 na 300
    if len(vzorcene_tocke) > max_vzorcne_tocke:
        indeksi = np.random.choice(len(vzorcene_tocke), max_vzorcne_tocke, replace=False)
        vzorcene_tocke = vzorcene_tocke[indeksi]
        print(f"Zmanjšano na {len(vzorcene_tocke)} vzorčnih točk")

    # Povečamo število iteracij
    max_iteracije = 10  # Povečano iz 3 na 10
    premaknjene_tocke = []

    # Vzorčimo podatke za računanje razdalj, a ne preagresivno
    max_tock_za_razdalje = 5000  # Povečano iz 1000 na 5000
    if len(znacilnice) > max_tock_za_razdalje:
        indeksi_razdalj = np.random.choice(len(znacilnice), max_tock_za_razdalje, replace=False)
        znacilnice_za_razdalje = znacilnice[indeksi_razdalj]
    else:
        znacilnice_za_razdalje = znacilnice
        indeksi_razdalj = np.arange(len(znacilnice))

    print("Računam premaknjene točke...")
    for i, tocka in enumerate(vzorcene_tocke):
        if i % 10 == 0:
            print(f"Procesiranje točke {i + 1}/{len(vzorcene_tocke)}")

        trenutna_tocka = tocka.copy()

        for it in range(max_iteracije):
            # Vektoriziran izračun razdalj na zmanjšanem vzorcu
            razdalje = np.sum(np.abs(znacilnice_za_razdalje - trenutna_tocka), axis=1)
            utezi = gaussovo_jedro(razdalje, velikost_okna)

            # Preveri če so utezi prenizke
            if np.sum(utezi) < 1e-10:
                break

            nova_tocka = np.sum(utezi[:, np.newaxis] * znacilnice_za_razdalje, axis=0) / np.sum(utezi)

            # Zaustavimo iteracije, če je premik zelo majhen
            if manhattanska_razdalja(nova_tocka, trenutna_tocka) < 0.1:
                break

            trenutna_tocka = nova_tocka

        premaknjene_tocke.append(trenutna_tocka)

    print("Računam centre...")
    # Manjša minimalna razdalja med centri za boljše zaznavanje majhnih segmentov
    min_cd = velikost_okna / 4  # Zmanjšano iz velikost_okna/2
    centri = []

    # Povečamo število premaknjenih točk za združevanje centrov
    max_premaknjene_tocke = 200  # Povečano iz 50 na 200
    if len(premaknjene_tocke) > max_premaknjene_tocke:
        indeksi = np.random.choice(len(premaknjene_tocke), max_premaknjene_tocke, replace=False)
        premaknjene_tocke_sample = [premaknjene_tocke[i] for i in indeksi]
    else:
        premaknjene_tocke_sample = premaknjene_tocke

    for tocka in premaknjene_tocke_sample:
        dodan = False
        for i, center in enumerate(centri):
            if manhattanska_razdalja(tocka, center) < min_cd:
                centri[i] = (centri[i] + tocka) / 2
                dodan = True
                break

        if not dodan:
            centri.append(tocka)

    if not centri:  # Preveri, če so centri prazni
        print("Ni najdenih centrov, vračam originalno sliko")
        return slika.copy()  # Vrni original, če ni centrov

    centri = np.array(centri)
    print(f"Število najdenih centrov: {len(centri)}")

    # Zvišamo maksimalno število centrov za boljšo ločljivost segmentov
    max_centri = 15  # Povečano iz 10 na 15
    if len(centri) > max_centri:
        print(f"Zmanjšanje centrov iz {len(centri)} na {max_centri}")
        try:
            # Uporabimo OpenCV k-means za združevanje centrov
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1.0)
            _, labels, novi_centri = cv.kmeans(
                centri.astype(np.float32),
                max_centri,
                None,
                criteria,
                10,
                cv.KMEANS_RANDOM_CENTERS
            )
            centri = novi_centri
        except Exception as e:
            print(f"Napaka pri združevanju centrov: {e}")
            # Če pride do napake, enostavno vzamemo prvih max_centri centrov
            centri = centri[:max_centri]

    print("Ustvarjam segmentirano sliko...")
    # Izločimo barve iz centrov
    if dimenzija == 3:
        barve_centrov = centri
    else:
        barve_centrov = centri[:, 2:5]

    # Direktno segmentiramo sliko
    segmentirana_slika = np.zeros((height, width, 3), dtype=np.uint8)

    # Za vsak piksel v originalni sliki poiščemo najbližji center
    for y in range(height):
        for x in range(width):
            piksel = slika[y, x].astype(np.float64)

            # Ustvarimo značilnico za piksel
            if dimenzija == 3:
                znacilnica = piksel
            else:
                x_norm = x * 255.0 / width
                y_norm = y * 255.0 / height
                znacilnica = np.concatenate(([x_norm, y_norm], piksel))

            # Poiščemo najbližji center
            najblizji_center = 0
            min_razdalja = float('inf')

            for i, center in enumerate(centri):
                razdalja = manhattanska_razdalja(znacilnica, center)
                if razdalja < min_razdalja:
                    min_razdalja = razdalja
                    najblizji_center = i

            # Dodelimo barvo pikslu
            if najblizji_center < len(barve_centrov):
                segmentirana_slika[y, x] = barve_centrov[najblizji_center]

    print("Mean-shift segmentacija končana")
    return segmentirana_slika.astype(np.uint8)

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
            # Simplified center initialization
            poskusi = 0
            centri = []

            while len(centri) < k and poskusi < 100:  # Dodaj omejitev poskusov
                poskusi += 1
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

            # Če nismo našli dovolj centrov, dodamo naključne
            while len(centri) < k:
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                centri.append(slika[y, x].astype(np.float64))

            centri = np.array(centri)

        elif dimenzija_centra == 5:  # barve + lokacije
            poskusi = 0
            centri = []

            while len(centri) < k and poskusi < 100:  # Dodaj omejitev poskusov
                poskusi += 1
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

            # Če nismo našli dovolj centrov, dodamo naključne
            while len(centri) < k:
                y = np.random.randint(0, h)
                x = np.random.randint(0, w)
                barva = slika[y, x].astype(np.float64)
                y_norm = y * 255.0 / h
                x_norm = x * 255.0 / w
                centri.append(np.concatenate(([x_norm, y_norm], barva)))

            centri = np.array(centri)

    return centri


def manhattanska_razdalja(vektor1, vektor2):
    """Izračuna manhattansko razdaljo med dvema vektorjema."""
    return np.sum(np.abs(vektor1 - vektor2))


def gaussovo_jedro(d, h):
    """Izračuna gaussovo jedro."""
    return np.exp(-d ** 2 / (2 * h ** 2))


def main():
    """Glavna funkcija za testiranje segmentacije slike."""
    # Nastavi poti za testne slike
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    test_dirs = [
        os.path.join(parent_dir, "testne_slike"),
        os.path.join(current_dir, "testne_slike"),
        "testne_slike"
    ]

    slika = None
    for test_dir in test_dirs:
        try:
            img_path = os.path.join(test_dir, "zelenjava.png")
            if os.path.exists(img_path):
                slika = cv.imread(img_path)
                if slika is not None:
                    print(f"Slika uspešno naložena iz: {img_path}")
                    break
        except Exception as e:
            print(f"Napaka pri nalaganju iz {test_dir}: {e}")

    if slika is None:
        print("Slika ni bila najdena. Poskušam z alternativnimi slikami...")
        # Try loading any image from the directory
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for file in os.listdir(test_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            img_path = os.path.join(test_dir, file)
                            slika = cv.imread(img_path)
                            if slika is not None:
                                print(f"Alternativna slika naložena: {img_path}")
                                break
                        except Exception:
                            continue
            if slika is not None:
                break

    if slika is None:
        # Create a small test image if no image could be loaded
        print("Ustvarjam testno sliko...")
        slika = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

    # Zmanjšaj velikost slike za hitrejše procesiranje
    max_dimension = 400
    h, w = slika.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        slika = cv.resize(slika, (int(w * scale), int(h * scale)))
        print(f"Slika zmanjšana na {slika.shape[1]}x{slika.shape[0]}")

    # Prikaži originalno sliko
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv.cvtColor(slika, cv.COLOR_BGR2RGB))
    plt.title("Originalna slika")
    plt.axis('off')

    try:
        # K-means segmentacija s samo enim parametrom za hitrejše testiranje
        print("Izvajam K-means segmentacijo...")
        seg_slika_kmeans = kmeans(slika, k=3, iteracije=5)
        plt.subplot(2, 2, 2)
        plt.imshow(cv.cvtColor(seg_slika_kmeans, cv.COLOR_BGR2RGB))
        plt.title("K-means: k=3")
        plt.axis('off')

        # Shrani in prikaži sproti, če pride do napake pri drugi metodi
        plt.tight_layout()
        plt.savefig("kmeans_rezultat.png")
        print("K-means rezultat shranjen v 'kmeans_rezultat.png'")
    except Exception as e:
        print(f"Napaka pri K-means segmentaciji: {e}")

    try:
        # Mean-shift segmentacija s samo enim parametrom za hitrejše testiranje
        print("Izvajam Mean-shift segmentacijo...")
        # Dodaj časovno omejitev
        import time
        start_time = time.time()
        seg_slika_ms = meanshift(slika, velikost_okna=30, dimenzija=3)
        end_time = time.time()
        print(f"Mean-shift končan v {end_time - start_time:.2f} sekundah")

        plt.subplot(2, 2, 3)
        plt.imshow(cv.cvtColor(seg_slika_ms, cv.COLOR_BGR2RGB))
        plt.title("Mean-shift: h=30, dim=3")
        plt.axis('off')

        # Shrani rezultat
        plt.tight_layout()
        plt.savefig("meanshift_rezultat.png")
        print("Mean-shift rezultat shranjen v 'meanshift_rezultat.png'")
    except Exception as e:
        print(f"Napaka pri Mean-shift segmentaciji: {e}")
        import traceback
        traceback.print_exc()

    # Prikaži rezultate
    try:
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Napaka pri prikazu rezultatov: {e}")
        print("Rezultati shranjeni v slikovne datoteke.")


if __name__ == "__main__":
    main()