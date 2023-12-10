import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.misc import ascent
from scipy.misc import face
from scipy.fftpack import dct, idct

# Functia pentru DCT 2D
def dct_2d(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

# Functia pentru inversarea DCT 2D
def idct_2d(image):
    return idct(idct(image.T, norm='ortho').T, norm='ortho')


# if __name__ == '__main__':
#     # Incarcam imaginea 'ascent'
#     img = ascent()
#
#     # Dimensiunile imaginii
#     h, w = img.shape
#
#     # Blocuri de 8x8
#     block_size = 8
#
#     # Matricea de cuantizare JPEG furnizata
#     Q_jpeg = np.array([
#         [16, 11, 10, 16, 24, 40, 51, 61],
#         [12, 12, 14, 19, 26, 28, 60, 55],
#         [14, 13, 16, 24, 40, 57, 69, 56],
#         [14, 17, 22, 29, 51, 87, 80, 62],
#         [18, 22, 37, 56, 68, 109, 103, 77],
#         [24, 35, 55, 64, 81, 104, 113, 92],
#         [49, 64, 78, 87, 103, 121, 120, 101],
#         [72, 92, 95, 98, 112, 100, 103, 99]
#     ])
#
#     # Procesam imaginea bloc cu bloc
#     jpeg_img = np.zeros_like(img, dtype=np.float32)
#     for i in range(0, h, block_size):
#         for j in range(0, w, block_size):
#             block = img[i:i + block_size, j:j + block_size]
#             dct_block = dct_2d(block)
#             quantized_block = np.round(dct_block / Q_jpeg) * Q_jpeg
#             idct_block = idct_2d(quantized_block)
#             jpeg_img[i:i + block_size, j:j + block_size] = idct_block
#
#     # Afisam rezultatele
#     plt.subplot(121)
#     plt.imshow(img, cmap=plt.cm.gray)
#     plt.title('Original')
#
#     plt.subplot(122)
#     plt.imshow(jpeg_img, cmap=plt.cm.gray)
#     plt.title('JPEG')
#
#     plt.show()
#
#     # Calculam componentele non-zero
#     y_nnz = np.count_nonzero(dct_2d(img[:8, :8]))
#     y_jpeg_nnz = np.count_nonzero(np.round(dct_2d(img[:8, :8]) / Q_jpeg) * Q_jpeg)
#
#     print('Componente în frecvență: ' + str(y_nnz) +
#           '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))




# Conversia din RGB in Y'CbCr
def rgb_to_ycbcr(img):
    Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    Cb = -0.1687 * img[:, :, 0] - 0.3313 * img[:, :, 1] + 0.5 * img[:, :, 2] + 128
    Cr = 0.5 * img[:, :, 0] - 0.4187 * img[:, :, 1] - 0.0813 * img[:, :, 2] + 128
    return np.stack([Y, Cb, Cr], axis=2)


# Conversia din Y'CbCr in RGB
def ycbcr_to_rgb(img):
    R = img[:, :, 0] + 1.402 * (img[:, :, 2] - 128)
    G = img[:, :, 0] - 0.344136 * (img[:, :, 1] - 128) - 0.714136 * (img[:, :, 2] - 128)
    B = img[:, :, 0] + 1.772 * (img[:, :, 1] - 128)
    return np.stack([R, G, B], axis=2).clip(0, 255).astype(np.uint8)

# Functia pentru aplicarea DCT pe un canal
def apply_dct_on_channel(channel, block_size, quant_matrix):
    channel_dct = np.zeros_like(channel, dtype=np.float32)
    for i in range(0, channel.shape[0], block_size):
        for j in range(0, channel.shape[1], block_size):
            block = channel[i:i+block_size, j:j+block_size]
            dct_block = dct_2d(block)
            quantized_block = np.round(dct_block / quant_matrix) * quant_matrix
            idct_block = idct_2d(quantized_block)
            channel_dct[i:i+block_size, j:j+block_size] = idct_block
    return channel_dct

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def adjust_quantization_matrix(matrix, current_mse, mse_threshold, adjustment_factor=1.5):
    # Creștem factorul de cuantizare doar dacă MSE-ul actual este sub prag
    # Aceasta va crește nivelul de compresie (și potențial MSE-ul)
    if current_mse < mse_threshold:
        return matrix * adjustment_factor
    return matrix


# if __name__ == '__main__':
#     # Incarcam imaginea color 'face'
#     img_color = face()
#
#     # Matricea de cuantizare JPEG
#     Q_jpeg = np.array([
#         [16, 11, 10, 16, 24, 40, 51, 61],
#         [12, 12, 14, 19, 26, 28, 60, 55],
#         [14, 13, 16, 24, 40, 57, 69, 56],
#         [14, 17, 22, 29, 51, 87, 80, 62],
#         [18, 22, 37, 56, 68, 109, 103, 77],
#         [24, 35, 55, 64, 81, 104, 113, 92],
#         [49, 64, 78, 87, 103, 121, 120, 101],
#         [72, 92, 95, 98, 112, 100, 103, 99]
#     ])
#
#     # Aplicam transformarea pe imaginea color
#     img_ycbcr = rgb_to_ycbcr(img_color)
#
#     # Dimensiunile imaginii Y'CbCr
#     h, w, _ = img_ycbcr.shape
#
#     # Blocuri de 8x8
#     block_size = 8
#
#     # Aplicam DCT, cuantizarea si inversarea DCT pe fiecare canal
#     jpeg_ycbcr = np.zeros_like(img_ycbcr, dtype=np.float32)
#     for i in range(3):
#         jpeg_ycbcr[:, :, i] = apply_dct_on_channel(img_ycbcr[:, :, i], block_size, Q_jpeg)
#
#     # Convertim inapoi in RGB
#     jpeg_rgb = ycbcr_to_rgb(jpeg_ycbcr)
#
#     # Afisam rezultatele
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img_color)
#     plt.title('Original RGB')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(jpeg_rgb)
#     plt.title('JPEG RGB')
#
#     plt.show()


# if __name__ == '__main__':
#     # Incarcam imaginea color 'face'
#     img_color = face()
#
#     # Pragul MSE specificat de utilizator
#     mse_threshold = 100  # Setează o valoare specifică aici
#
#     # Matricea de cuantizare JPEG inițială
#     Q_jpeg = np.array([
#         [16, 11, 10, 16, 24, 40, 51, 61],
#         [12, 12, 14, 19, 26, 28, 60, 55],
#         [14, 13, 16, 24, 40, 57, 69, 56],
#         [14, 17, 22, 29, 51, 87, 80, 62],
#         [18, 22, 37, 56, 68, 109, 103, 77],
#         [24, 35, 55, 64, 81, 104, 113, 92],
#         [49, 64, 78, 87, 103, 121, 120, 101],
#         [72, 92, 95, 98, 112, 100, 103, 99]
#     ])
#
#     # Convertim imaginea inițială în Y'CbCr
#     img_ycbcr = rgb_to_ycbcr(img_color)
#
#     # Dimensiunile și blocurile
#     h, w, _ = img_ycbcr.shape
#     block_size = 8
#
#     # Inițializăm MSE-ul și matricea de cuantizare ajustată
#     current_mse = 0
#     adjusted_quant_matrix = Q_jpeg.copy()
#
#     while current_mse < mse_threshold:
#         # Aplicăm procesul JPEG pe fiecare canal
#         jpeg_ycbcr = np.zeros_like(img_ycbcr, dtype=np.float32)
#
#         for i in range(3):
#             jpeg_ycbcr[:, :, i] = apply_dct_on_channel(img_ycbcr[:, :, i], block_size, adjusted_quant_matrix)
#
#         # Convertim în RGB pentru a calcula MSE
#         jpeg_rgb = ycbcr_to_rgb(jpeg_ycbcr)
#         current_mse = calculate_mse(img_color, jpeg_rgb)
#
#         # Ajustăm matricea de cuantizare
#         adjusted_quant_matrix = adjust_quantization_matrix(adjusted_quant_matrix, current_mse, mse_threshold)
#
#     # Afișăm rezultatele
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img_color)
#     plt.title('Original RGB')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(jpeg_rgb)
#     plt.title('JPEG RGB cu MSE Ajustat')
#
#     plt.show()



def process_frame(frame, block_size, quant_matrix):
    """Procesează un singur cadru: RGB -> Y'CbCr -> DCT -> IDCT -> RGB."""
    frame_ycbcr = rgb_to_ycbcr(frame)
    compressed_frame = np.zeros_like(frame_ycbcr)

    # Aplică DCT și IDCT pe fiecare canal
    for i in range(3):
        compressed_frame[:, :, i] = apply_dct_on_channel(frame_ycbcr[:, :, i], block_size, quant_matrix)

    return ycbcr_to_rgb(compressed_frame)


if __name__ == '__main__':
    # Deschide clipul video
    video = cv2.VideoCapture('C:/Users/Vlad/Downloads/people.mp4')

    block_size = 8

    Q_jpeg = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 28, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Procesează cadrul
        compressed_frame = process_frame(frame, block_size, Q_jpeg)

        # Afișează cadrul comprimat (sau salvează-l)
        cv2.imshow('Compressed Frame', compressed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()








