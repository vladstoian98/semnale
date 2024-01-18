from PIL import Image
from scipy.misc import face
import numpy as np

# def redimensioneaza_si_afiseaza_imagine(noua_latime, noua_inaltime):
#     # Încărcarea unei imagini de exemplu din biblioteca Pillow
#     img = Image.fromarray(face())
#
#     # Redimensionează imaginea
#     img_redimensionata = img.resize((noua_latime, noua_inaltime))
#
#     # Afisează imaginea redimensionată
#     img_redimensionata.show()
#
# if __name__ == '__main__':
#
#     # Exemplu de utilizare
#     noua_latime = 1600  # Înlocuiește cu lățimea dorită
#     noua_inaltime = 6000  # Înlocuiește cu înălțimea dorită
#
#     redimensioneaza_si_afiseaza_imagine(noua_latime, noua_inaltime)

##############################################################

def nearest_neighbor_downsample(image, new_width, new_height):
    """
    Redimensionează o imagine folosind interpolarea celor mai apropiați vecini (downsampling).

    :param image: Imaginea originală reprezentată ca o listă 2D de valori ale pixelilor.
    :param new_width: Lățimea imaginii redimensionate.
    :param new_height: Înălțimea imaginii redimensionate.
    :return: Imaginea redimensionată ca o listă 2D de valori ale pixelilor.
    """
    original_height = len(image)
    original_width = len(image[0])

    # Calcularea raportului dintre dimensiunile noi și cele originale
    width_ratio = original_width / new_width
    height_ratio = original_height / new_height

    # Crearea noii imagini cu dimensiunile specificate
    downsampled_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    for i in range(new_height):
        for j in range(new_width):
            # Găsirea celui mai apropiat vecin din imaginea originală
            orig_i = int(i * height_ratio)
            orig_j = int(j * width_ratio)
            downsampled_image[i][j] = image[orig_i][orig_j]

    return downsampled_image


def nearest_neighbor_upsample(image, new_width, new_height):
    """
    Mărește dimensiunea unei imagini folosind interpolarea celor mai apropiați vecini (upsampling).

    :param image: Imaginea originală reprezentată ca o listă 2D de valori ale pixelilor.
    :param new_width: Lățimea imaginii mărite.
    :param new_height: Înălțimea imaginii mărite.
    :return: Imaginea mărită ca o listă 2D de valori ale pixelilor.
    """
    original_height = len(image)
    original_width = len(image[0])

    # Calcularea raportului dintre dimensiunile noi și cele originale
    width_ratio = original_width / new_width
    height_ratio = original_height / new_height

    # Crearea noii imagini cu dimensiunile specificate
    upsampled_image = [[0 for _ in range(new_width)] for _ in range(new_height)]

    for i in range(new_height):
        for j in range(new_width):
            # Găsirea celui mai apropiat vecin din imaginea originală
            orig_i = int(i * height_ratio)
            orig_j = int(j * width_ratio)
            upsampled_image[i][j] = image[orig_i][orig_j]

    return upsampled_image


def print_image(image):
    """
    Afișează imaginea reprezentată ca o listă 2D de valori ale pixelilor.

    :param image: Imaginea care va fi afișată, reprezentată ca o listă 2D.
    """
    for row in image:
        print(" ".join(str(pixel) for pixel in row))

# if __name__ == '__main__':
#     example_image = [
#         [0, 0, 0, 0, 0],
#         [0, 1, 1, 1, 0],
#         [0, 1, 0, 1, 0],
#         [0, 1, 1, 1, 0],
#         [0, 0, 0, 0, 0]
#     ]
#
#     print("Imaginea originala:")
#     print_image(example_image)
#
#     downsampled_image = nearest_neighbor_downsample(example_image, 4, 4)
#     print("\nImaginea downsampled:")
#     print_image(downsampled_image)
#
#     upsampled_image = nearest_neighbor_upsample(example_image, 7, 7)
#     print("\nImaginea upsampled:")
#     print_image(upsampled_image)


############################################################################

if __name__ == '__main__':
    # Crearea unei matrice de imagine de test 4x4
    image_matrix = np.array([[10, 20, 30, 40],
                             [50, 60, 70, 80],
                             [90, 100, 110, 120],
                             [130, 140, 150, 160]])


    # Functia pentru interpolare bilineara pentru subesantionare
    def subesantionare_bilineara(image_matrix, scale_factor):
        height, width = image_matrix.shape
        new_height = int(height / scale_factor)
        new_width = int(width / scale_factor)
        result_matrix = np.zeros((new_height, new_width), dtype=np.uint8)

        # Iteram prin matricea rezultat
        for i in range(new_height):
            for j in range(new_width):
                # Calculam pozitia (x, y) in matricea originala
                x = int(i * scale_factor)
                y = int(j * scale_factor)

                # Calculam diferenta dintre pozitia (x, y) calculata si pozitia (x, y) reala;
                # ne ajuta sa intelegem distanta de la pozitia calculata in poza originala la pozitia din poza redimensionata
                x_diff = (i * scale_factor) - x
                y_diff = (j * scale_factor) - y

                print(x_diff)

                if x + 1 < height and y + 1 < width:
                    # Calculeaza valoarea interpolata biliniar folosind cele patru puncte vecine
                    interpolated_value = (1 - x_diff) * (1 - y_diff) * image_matrix[x, y] + \
                                         (x_diff) * (1 - y_diff) * image_matrix[x + 1, y] + \
                                         (1 - x_diff) * (y_diff) * image_matrix[x, y + 1] + \
                                         (x_diff) * (y_diff) * image_matrix[x + 1, y + 1]
                    result_matrix[i, j] = int(interpolated_value)

        return result_matrix


    # Functia pentru interpolare bilineara pentru supesantionare
    def supesantionare_bilineara(image_matrix, scale_factor):
        height, width = image_matrix.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        result_matrix = np.zeros((new_height, new_width), dtype=np.uint8)

        # Iteram prin matricea rezultat
        for i in range(new_height):
            for j in range(new_width):
                # Calculam pozitia (x, y) in matricea rezultat
                x = i / scale_factor
                y = j / scale_factor

                # Determinam coordonatele celor patru puncte vecine in matricea originala
                x1, y1 = int(x), int(y)  # Coordonatele pixelului de sus-stanga
                x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)  # Coordonatele pixelului de jos-dreapta

                # Calculam diferenta dintre pozitia (x, y) calculata si pozitia (x1, y1) reala
                x_diff, y_diff = x - x1, y - y1

                # Calculeaza valoarea interpolata biliniar folosind cele patru puncte vecine
                interpolated_value = (1 - x_diff) * (1 - y_diff) * image_matrix[x1, y1] + \
                                     x_diff * (1 - y_diff) * image_matrix[x2, y1] + \
                                     (1 - x_diff) * y_diff * image_matrix[x1, y2] + \
                                     x_diff * y_diff * image_matrix[x2, y2]

                result_matrix[i, j] = int(interpolated_value)

        return result_matrix


    # Subesantionare cu un factor de 2
    subesantionare_factor = 2
    subesantionare_result = subesantionare_bilineara(image_matrix, subesantionare_factor)

    # Supesantionare cu un factor de 2
    supesantionare_factor = 2
    supesantionare_result = supesantionare_bilineara(image_matrix, supesantionare_factor)

    # Afisarea matricei originale si a rezultatelor
    print("Imaginea originala:")
    print(image_matrix)

    print("\nSubesantionare Bilineara:")
    print(subesantionare_result)

    print("\nSupesantionare Bilineara:")
    print(supesantionare_result)