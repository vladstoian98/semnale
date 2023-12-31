import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import numpy
import matplotlib.pyplot as plt
import time
from scipy import misc, ndimage
import numpy as np
import sounddevice as sd
from numpy.fft import ifft, fft, fftshift, fft2, ifft2, ifftshift
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import sawtooth, butter, filtfilt, cheby1
import pandas as pd
from skimage.restoration import estimate_sigma, denoise_nl_means
import cv2
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

#Laborator 1

def fct_x(t):
    return numpy.cos((520 * numpy.pi * t) + (numpy.pi / 3))

def fct_y(t):
    return numpy.cos((280 * numpy.pi * t) - (numpy.pi / 3))

def fct_z(t):
    return numpy.cos((120 * numpy.pi * t) + (numpy.pi / 3))

def sinus(a, frecventa, timp, faza):
    return a * numpy.sin(2 * numpy.pi * frecventa * timp + faza)

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     #lab1
#     #ex1 a b c
#     rata = numpy.arange(0, 0.03, 0.0005)
#
#     figura1, axe_grafic = plt.subplots(3)
#     figura1.suptitle('1b&c')
#
#     axe_grafic[0].plot(rata, fct_x(rata))
#     axe_grafic[1].plot(rata, fct_y(rata))
#     axe_grafic[2].plot(rata, fct_z(rata))
#
#     rata2 = numpy.arange(0, 0.03, 0.03/6)
#     print(rata2)
#
#     print(fct_x(rata2))
#     print(fct_y(rata2))
#     print(fct_z(rata2))
#
#     axe_grafic[0].plot(rata2, fct_x(rata2), 'o')
#     axe_grafic[1].plot(rata2, fct_y(rata2), 'o')
#     axe_grafic[2].plot(rata2, fct_z(rata2), 'o')
#
#     plt.show()



# if __name__ == '__main__':
#     #ex2a
#     rata = numpy.arange(0, 0.05, 0.000005)
#     timp = numpy.linspace(0, 0.05, 1600)
#
#     print(rata)
#     print(timp)
#
#     figura1, axa_grafic = plt.subplots(1)
#     figura1.suptitle('2a')
#
#     axa_grafic.plot(rata, sinus(1, 400, rata, 0))
#     axa_grafic.plot(timp, sinus(1, 400, timp, 0), 'o')
#
#     plt.show()

# if __name__ == '__main__':
#     #ex2b
#     timp = numpy.linspace(0, 3, 301)
#
#     plt.plot(time, sinus(1, 800, timp, 0))
#     plt.show()

# if __name__ == '__main__':
#     #ex2c
#     def sawtooth(t):
#         return numpy.mod(t, 1. / 240)
#
#
#     timp = numpy.linspace(0, 0.05, 500)
#
#     semnal = plt.plot(timp, sawtooth(timp))
#     plt.show()

# if __name__ == '__main__':
#     #ex2d
#     timp = numpy.linspace(0, 1, 51)
#
#     print(numpy.sign(sinus(1, 300, timp, 0)))
#
#     semnal = plt.plot(timp, numpy.sign(sinus(1, 300, timp, 0)))
#     plt.show()

# if __name__ == '__main__':
#     #ex2e
#     array = numpy.random.rand(128, 128)
#
#     plt.imshow(array)
#     plt.show()

def s_sin(amplitudine, frecv, timp, faza):
    return amplitudine * numpy.sin(2 * numpy.pi * frecv * timp + faza)

# if __name__ == '__main__':
#     # ex2f
#     matr = numpy.zeros((128, 128))
#
#     timp = numpy.linspace(0, 0.03, 128)
#
#     sinValori = (s_sin(1, 100, timp, 0) + 1) * 64
#
#     sinValoriIndici = sinValori.astype(int)
#
#     for i, y in enumerate(sinValoriIndici):
#         if 0 <= y < 128:
#             matr[y, i] = 1
#
#     plt.imshow(matr, cmap='gray')
#     plt.show()












#Laborator 2


def s_sin(amplitudine, frecv, timp, faza):
    return amplitudine * numpy.sin(2 * numpy.pi * frecv * timp + faza)

def s_cos(amplitudine, frecv, timp, faza):
    return amplitudine * numpy.cos(2 * numpy.pi * frecv * timp + faza)

# if __name__ == '__main__':
#     #Ex1
#     timp = numpy.linspace(0, 0.05, 500)
#
#     plt.plot(timp, s_sin(1, 400, timp, 0))
#     plt.title('sinus')
#     plt.show()
#
#     plt.plot(timp, s_cos(1, 400, timp, 0 - numpy.pi / 2))
#     plt.title('cosinus')
#     plt.show()

#ex2
# def adaugareDist(semnal, snr):
#     putereSemnal = numpy.mean(semnal ** 2)
#
#     putereDist = putereSemnal / snr
#
#     dist = numpy.sqrt(putereDist) * numpy.random.randn(*semnal.shape)
#
#     semnalDist = semnal + dist
#
#     return semnalDist


# if __name__ == '__main__':
#     #Ex2
#     p1 = 0
#     p2 = 1
#     p3 = 2
#     p4 = 4
#
#     snr = [0.1, 1, 10, 100]
#
#     timp = numpy.linspace(0, 0.05, 500)
#
#     fig, axs = plt.subplots(4)
#     axs[0].plot(timp, adaugareDist(s_sin(1, 400, timp, p1), snr[0]))
#     axs[1].plot(timp, adaugareDist(s_sin(1, 400, timp, p2), snr[1]))
#     axs[2].plot(timp, adaugareDist(s_sin(1, 400, timp, p3), snr[2]))
#     axs[3].plot(timp, adaugareDist(s_sin(1, 400, timp, p4), snr[3]))
#     plt.show()


# if __name__ == '__main__':
#     #ex3
#     rata = numpy.linspace(0, 10, 16000 * 10)
#
#     semnal = sinus(1, 4000, rata, 0)
#
#     sd.play(semnal, 16000)
#     sd.wait()

# if __name__ == '__main__':
#     #Ex 4
#     timp = numpy.linspace(0, 1, 50, endpoint=False)
#     sinSemnal = sinus(1, 440, timp, 0)
#     sawSemnal = sawtooth(2 * numpy.pi * 440 * timp)
#
#     semnalTotal = sinSemnal + sawSemnal
#
#     print(numpy.sum(semnalTotal))
#
#     fig, axs = plt.subplots(3)
#     axs[0].plot(timp, sinSemnal)
#     axs[1].plot(timp, sawSemnal)
#     axs[2].plot(timp, semnalTotal)
#
#     plt.show()

# if __name__ == '__main__':
#     #Ex5
#     rata1 = numpy.linspace(0, 5, 16000)
#     rata2 = numpy.linspace(0, 5, 16000)
#
#     semnal1 = sinus(1, 4000, rata1, 0)
#     semnal2 = sinus(1, 8000, rata2, 0)
#
#     semnalTotal = numpy.concatenate((semnal1, semnal2))
#
#     sd.play(semnalTotal, 16000)
#     sd.wait()
#     #sunetul produs de cele doua semnale este diferit

# if __name__ == '__main__':
#     #Ex6
#     timp = numpy.linspace(0, 0.05, 500)
#
#     semnal1 = sinus(1, 500 / 2, timp, 0)
#     semnal2 = sinus(1, 500 / 4, timp, 0)
#     semnal3 = sinus(1, 0, timp, 0)
#
#     fig, axs = plt.subplots(3)
#     axs[0].plot(timp, semnal1)
#     axs[1].plot(timp, semnal2)
#     axs[2].plot(timp, semnal3)
#
#     plt.show()
#     #cu cat frecventa este mai mica, cu atat mai putine "cicluri" (bump-uri) sunt

# if __name__ == '__main__':
#     #Ex7
#     timp = numpy.linspace(0, 0.05, 500)
#     timpDecimat1 = timp[::4]
#     timpDecimat2 = timp[1::4]
#
#     semnal1 = sinus(1, 1000, timp, 0)
#     semnal2 = sinus(1, 1000, timpDecimat1, 0)
#     semnal3 = sinus(1, 1000, timpDecimat2, 0)
#
#     fig, axs = plt.subplots(3)
#     axs[0].plot(timp, semnal1)
#     axs[1].plot(timpDecimat1, semnal2)
#     axs[2].plot(timpDecimat2, semnal3)
#
#     plt.show()
#     #Semnalul 3 arata ca semnalul 2 inversat

# if __name__ == '__main__':
#     #Ex 8
#     interval = numpy.linspace(-numpy.pi/2, numpy.pi/2, 1000)
#
#     sinInterval = numpy.sin(interval)
#
#     aproximarePade = (interval - (7 * interval ** 3) / 60) / (1 + (interval ** 2) / 20)
#
#     fig, axs = plt.subplots(3)
#     axs[0].plot(interval, interval)
#     axs[1].plot(interval, aproximarePade)
#     axs[2].plot(interval, sinInterval)
#
#     for ax in axs:
#         ax.set_yscale("symlog") #symlog sau log? daca folosim log valorile <0 nu vor fi afisate
#
#     plt.show()



#Laborator 3
# if __name__ == '__main__':
#     #Ex1
#     N = 6
#     F = numpy.zeros((N, N), dtype=complex)
#
#     for m in range(N):
#         for k in range(N):
#             x = -2j * numpy.pi * m * k / N
#             F[m, k] = numpy.exp(x)
#
#     print(F)
#
#     plt.figure(figsize=(15, 40))
#     for m in range(N):
#         plt.subplot(N, 2, 2 * m + 1)
#         plt.plot(numpy.real(F[m, :]))
#         plt.title(f"Linia {m + 1} - real")
#
#         plt.subplot(N, 2, 2 * m + 2)
#         plt.plot(numpy.imag(F[m, :]))
#         plt.title(f"Linia {m + 1} - imaginar")
#
#     plt.tight_layout()
#     plt.show()
#
#     FConj = numpy.conjugate(F.T)  # Matricea conjugată transpusă (hermitiană)
#     matriceIdentitate = numpy.dot(F, FConj)
#     # matriceIdentitate ar tb sa aiba 1 pe diagonala si 0 in rest
#     unitar = numpy.allclose(matriceIdentitate, numpy.eye(N) * N, atol=1e-10)
#
#     if unitar:
#         print("Matricea Fourier este unitară.")
#     else:
#         print("Matricea Fourier NU este unitară.")


# if __name__ == '__main__':
#     #Ex2
#     nrPuncte = 1000
#
#     matr = numpy.linspace(0, 1, nrPuncte)
#
#     print(matr)
#
#     semnalSin = numpy.sin(2 * 5 * numpy.pi * matr)  # frecventa 5
#
#     y = semnalSin * numpy.exp(-2j * numpy.pi * 7 * matr) # omega este 7 in acest caz, pentru a reprezenta graficul 4
#     # din figura 2; pt graficul 2 din figura 1 trebuie sa alegem omega = 1
#     # daca frecventa si omega sunt egale se va afisa un cerc
#
#     r = y.real
#     i = y.imag
#
#     plt.plot(r, i)
#     plt.xlim(-1, 1)
#     plt.ylim(-1, 1)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

# if __name__ == '__main__':
#     #Ex3
#     N = 1000
#     t = numpy.linspace(0, 1, N, endpoint=False)
#     f1, f2, f3 = 20, 30, 100
#
#     semnaleConcatenate = numpy.sin(2 * numpy.pi * f1 * t) + numpy.sin(2 * numpy.pi * f2 * t) + numpy.sin(2 * numpy.pi * f3 * t)
#
#
#     vect = numpy.zeros(N, complex)
#     for k in range(N):
#         for n in range(N):
#             vect[k] += semnaleConcatenate[n] * numpy.exp(-2j * numpy.pi * k * n / N)
#
#     modul = numpy.abs(vect)
#
#     plt.plot(t, semnaleConcatenate)
#     plt.title("Semnal original")
#     plt.show()
#
#     plt.subplot(2, 1, 2)
#     plt.plot(numpy.linspace(0, N // 2, N // 2), modul[:N // 2])
#     plt.tight_layout()
#     plt.show()





# Lab 4
# if __name__ == '__main__':
#     #Ex1
#     N = [128, 256, 512, 1024, 2048]
#     timpiTrecutiProprii = []
#     timpiTrecutiPython = []
#
#     for i in N:
#         t = numpy.linspace(0, 1, i, endpoint=False)
#         f1, f2, f3 = 20, 30, 100
#         semnaleConcatenate = numpy.sin(2 * numpy.pi * f1 * t) + numpy.sin(2 * numpy.pi * f2 * t) + numpy.sin(2 * numpy.pi * f3 * t)
#
#         fftPropriuT1 = time.perf_counter()
#
#         vect = numpy.zeros(i, complex)
#
#         for k in range(i):
#             for n in range(i):
#                 vect[k] += semnaleConcatenate[n] * numpy.exp(-2j * numpy.pi * k * n / i)
#
#         fftPropriuT2 = time.perf_counter()
#         timpPropriu = fftPropriuT2 - fftPropriuT1
#         timpiTrecutiProprii.append(timpPropriu)
#
#
#     for i in N:
#         t = numpy.linspace(0, 1, i, endpoint=False)
#         f1, f2, f3 = 20, 30, 100
#         semnaleConcatenate = numpy.sin(2 * numpy.pi * f1 * t) + numpy.sin(2 * numpy.pi * f2 * t) + numpy.sin(2 * numpy.pi * f3 * t)
#
#         fftPythonT1 = time.perf_counter()
#
#         fftVar = numpy.fft.fft(semnaleConcatenate)
#
#         fftPythonT2 = time.perf_counter()
#
#         timpPython = fftPythonT2 - fftPythonT1
#         timpiTrecutiPython.append(timpPython)
#
#     timpiTrecutiPropriiMs = [x * 1000 for x in timpiTrecutiProprii]
#     timpiTrecutiPythonMs = [x * 1000 for x in timpiTrecutiPython]
#
#     print(timpiTrecutiPython)
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(N, timpiTrecutiPropriiMs, 'o-', color='red')
#     plt.plot(N, timpiTrecutiPythonMs, 'o-', color='green')
#     plt.yscale('log')
#     plt.xlabel('Size of N')
#     plt.ylabel('Elapsed Time (milliseconds)')
#     plt.grid(True, which="both", ls="--")
#
#     plt.show()

# if __name__ == '__main__':
#     #Ex2
#     t_samples = numpy.linspace(0, 1, 10, endpoint=False)
#     t = numpy.linspace(0, 1, 1000, endpoint=False)
#
#     signal_original = numpy.sin(2 * numpy.pi * 30 * t)
#     signal_alias1 = numpy.sin(2 * numpy.pi * 20 * t)
#     signal_alias2 = numpy.sin(2 * numpy.pi * 10 * t)
#
#     samples_original = numpy.sin(2 * numpy.pi * 30 * t_samples)
#     samples_alias1 = numpy.sin(2 * numpy.pi * 20 * t_samples)
#     samples_alias2 = numpy.sin(2 * numpy.pi * 10 * t_samples)
#
#     plt.figure(figsize=(10, 8))
#
#     plt.subplot(3, 1, 1)
#     plt.plot(t, signal_original)
#     plt.scatter(t_samples, samples_original)
#
#     plt.subplot(3, 1, 2)
#     plt.plot(t, signal_alias1)
#     plt.scatter(t_samples, samples_alias1)
#
#     plt.subplot(3, 1, 3)
#     plt.plot(t, signal_alias2)
#     plt.scatter(t_samples, samples_alias2)
#
#     plt.tight_layout()
#     plt.show()

# if __name__ == '__main__':
#     #Ex3
#     t_samples = numpy.linspace(0, 1, 25, endpoint=False)
#     t = numpy.linspace(0, 1, 1000, endpoint=False)
#
#     signal_original = numpy.sin(2 * numpy.pi * 75 * t)
#     signal_alias1 = numpy.sin(2 * numpy.pi * 50 * t)
#     signal_alias2 = numpy.sin(2 * numpy.pi * 25 * t)
#
#     samples_original = numpy.sin(2 * numpy.pi * 5 * t_samples)
#     # samples_alias1 = numpy.sin(2 * numpy.pi * 50 * t_samples)
#     # samples_alias2 = numpy.sin(2 * numpy.pi * 25 * t_samples)
#
#     plt.figure(figsize=(10, 8))
#
#     plt.subplot(3, 1, 1)
#     plt.plot(t, signal_original)
#     plt.scatter(t_samples, samples_original)
#
#     plt.subplot(3, 1, 2)
#     plt.plot(t, signal_alias1)
#     plt.scatter(t_samples, samples_original)
#
#     plt.subplot(3, 1, 3)
#     plt.plot(t, signal_alias2)
#     plt.scatter(t_samples, samples_original)
#
#     plt.tight_layout()
#     plt.show()

    #Ex 4: 400 hz

    #Ex7: 10 dB

#Lab5

# if __name__ == '__main__':
#     file_path = 'C:/Users/Vlad/Downloads/Train.csv'
#
#     df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(df.index, df['Count'])
#     plt.title('trafic')
#     plt.xlabel('timp')
#     plt.ylabel('count')
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

#a
# semnalul a fost eșantionat cu o frecvență de 1 eșantion pe oră,
# deoarece pentru a obține 18288 eșantioane, cu măsurători luate oră de oră,
# ar însemna că datele acoperă 18288 de ore.
# Deci, frecvența de eșantionare este de 1 Hz,
# ceea ce înseamnă un eșantion pe oră.

#b
#18288/24 = 762 zile

#c
#Dacă semnalul a fost eșantionat corect și optim,
# atunci frecvența de eșantionare este de două ori
# mai mare decât frecvența maximă a semnalului.
# Astfel, deoarece frecventa de esantionare este 1 Hz,


#d
# if __name__ == '__main__':
#     file_path = 'C:/Users/Vlad/Downloads/Train.csv'
#
#     df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
#
#     time_interval = (df.index[1] - df.index[0]).total_seconds()
#
#     print(time_interval)
#
#     f_sample = 1 / time_interval
#
#     s = df['Count'].values
#
#     fft_result = numpy.fft.fft(s)
#
#     n = s.size
#     freq = numpy.fft.fftfreq(n, d=time_interval)
#
#     print(n, freq)
#
#     fft_positive_half = fft_result[:n // 2]
#     freq_positive_half = freq[:n // 2]
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(freq_positive_half, numpy.abs(fft_positive_half))
#     plt.title('Magnitudinea Transformatei Fourier a Numărului de Trafic')
#     plt.xlabel('Frecvență (Hz)')
#     plt.ylabel('Magnitudine')
#     plt.grid(True)
#     plt.show()

#e
# Prima valoare în vectorul rezultat din FFT (fft_result[0]) corespunde componentei continue. Aceasta reprezintă componenta continua a semnalului. Pentru a scoate componenta continua, facem fft_result[0] = 0
# if __name__ == '__main__':
#     file_path = 'C:/Users/Vlad/Downloads/Train.csv'
#
#     df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
#
#     time_interval = (df.index[1] - df.index[0]).total_seconds()
#
#     f_sample = 1 / time_interval
#
#     s = df['Count'].values
#
#     fft_result = numpy.fft.fft(s)
#
#     n = s.size
#     freq = numpy.fft.fftfreq(n, d=time_interval)
#
#     cont_comp = fft_result[0]
#
#     print(numpy.abs(cont_comp))
#
#     # Remove the DC component if it is present
#     if numpy.abs(cont_comp) > 0:
#         fft_result[0] = 0
#
#     fft_positive_half = fft_result[:n // 2]
#     freq_positive_half = freq[:n // 2]
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(freq_positive_half, numpy.abs(fft_positive_half))
#     plt.title('Magnitudinea Transformatei Fourier a Numărului de Trafic')
#     plt.xlabel('Frecvență (Hz)')
#     plt.ylabel('Magnitudine')
#     plt.grid(True)
#     plt.show()

#f
# Magnitudinile maxime ar trebui sa fie reprezentate de orele de varf din ziua respectiva
# if __name__ == '__main__':
#     file_path = 'C:/Users/Vlad/Downloads/Train.csv'
#
#     df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
#
#     time_interval = (df.index[1] - df.index[0]).total_seconds()
#
#     # print(time_interval)
#
#     f_sample = 1 / time_interval
#
#     s = df['Count'].values
#
#     fft_result = numpy.fft.fft(s)
#
#     n = s.size
#     freq = numpy.fft.fftfreq(n, d=time_interval)
#
#     #Calculam modulul
#     modul = numpy.abs(fft_result)
#
#     max = modul.argsort()[-5:-1]
#
#     top_magnitudes = modul[max]
#     top_frequencies = freq[max]
#
#     for i in range(4):
#         print(f"Frecvența {top_frequencies[i]} Hz are o magnitudine de {top_magnitudes[i]}")


# g
# if __name__ == '__main__':
#     file_path = 'C:/Users/Vlad/Downloads/Train.csv'
#
#     df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
#
#     esantionIncepere = df[df['Count'] > 1000].first_valid_index()
#
#     while esantionIncepere.weekday() != 0:
#         esantionIncepere += pd.Timedelta(days=1)
#
#     final = esantionIncepere + pd.DateOffset(months=1)
#     traficLuna = df[esantionIncepere:final]
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(traficLuna.index, traficLuna['Count'])
#     plt.xlabel('Time')
#     plt.ylabel('Count')
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()


#i
# if __name__ == '__main__':
#     file_path = 'C:/Users/Vlad/Downloads/Train.csv'
#
#     df = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
#
#     start_date = df.index.min()
#     end_date = start_date + pd.Timedelta(days=3)
#
#     treiZile = df[(df.index >= start_date) & (df.index < end_date)]
#
#     ordin = 5  # ordin mai mare rezulta intr-un "slope" mai brusc
#     cutoff_frequency = 0.1  # fractiune din frecventa Nyquist
#
#     b, a = butter(ordin, cutoff_frequency, btype='low', analog=False)
#
#     treiZile['Filtered_Count'] = filtfilt(b, a, treiZile['Count'])
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(treiZile.index, treiZile['Count'], label='Original')
#     plt.plot(treiZile.index, treiZile['Filtered_Count'], label='Filtered', color='red')
#     plt.xlabel('Time')
#     plt.ylabel('Count')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()


#Lab 6

#Ex1
# if __name__ == '__main__':
#     numpy.random.seed(0)
#
#     N = 100
#     x = numpy.random.rand(N)
#
#     vectors = [x]
#
#     for _ in range(3):
#         x = x * x
#         vectors.append(x)
#
#     plt.figure(figsize=(14, 7))
#     for i, vector in enumerate(vectors):
#         plt.subplot(2, 2, i + 1)
#         plt.plot(vector)
#         plt.title(f'Iterarea {i}')
#     plt.tight_layout()
#     plt.show()

#Ex2
# if __name__ == '__main__':
#     # Acum vor fi aceleasi date generate
#     numpy.random.seed(0)
#
#     # Gradul maxim
#     N = 10
#
#     # Generam coeficientii
#     p_coefficients = numpy.random.randint(-10, 10, N + 1)
#     q_coefficients = numpy.random.randint(-10, 10, N + 1)
#
#     # Produs direct
#     r_classic = numpy.polynomial.polynomial.polymul(p_coefficients, q_coefficients)
#
#     # asiguram prin 0 padding ca vectorii de coeficienti sunt indeajuns de lungi pt a reprezenta toti termenii posibili in polinomul rezultat
#     p_padded = numpy.pad(p_coefficients, (0, len(q_coefficients) - 1))
#     q_padded = numpy.pad(q_coefficients, (0, len(p_coefficients) - 1))
#     r_fft = ifft(fft(p_padded) * fft(
#         q_padded)).real
#
#     r_fft = numpy.round(r_fft)
#
#     print("Coeficientii lui p(x):", p_coefficients)
#     print("Coeficientii lui q(x):", q_coefficients)
#     print("Produs clasic:", r_classic)
#     print("Produs cu FFT:", r_fft)


#Ex3
# if __name__ == '__main__':
#     t = numpy.arange(0, 1, 1 / 5000)
#
#     sinusul = numpy.sin(2 * numpy.pi * 100 * t)
#
#     # Dimensiune fereastra
#     dimFer = 200
#
#     # Aplicam fereastra dreptunghiulara
#     fereastraDrept = numpy.ones(dimFer)
#     sinFerDrept = sinusul[:dimFer] * fereastraDrept
#
#     print("Fereastra drept: ", fereastraDrept)
#
#     # Aplicam fereastra hanning
#     fereastraHanning = numpy.hanning(dimFer)
#     sinFerHanning = sinusul[:dimFer] * fereastraHanning
#
#     print("Fereastra drept: ", fereastraHanning)
#
#     plt.figure(figsize=(14, 7))
#
#     # Sin Original
#     plt.subplot(2, 1, 1)
#     plt.plot(t[:dimFer], sinusul[:dimFer])
#     plt.title('Sin Original')
#
#     # Sin dupa fereastra drept
#     plt.subplot(2, 2, 3)
#     plt.plot(t[:dimFer], sinFerDrept)
#     plt.title('Sin dupa fereastra drept')
#
#     # Sin dupa fereastra hanning
#     plt.subplot(2, 2, 4)
#     plt.plot(t[:dimFer], sinFerHanning)
#     plt.title('Sin dupa fereastra hanning')
#
#     plt.tight_layout()
#     plt.show()

#Ex4
#a,b
# if __name__ == '__main__':
#     df = pd.read_csv('C:/Users/Vlad/Downloads/Train.csv')
#
    # df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
    #
    # start_date = df['Datetime'].min()
    # end_date = start_date + pd.Timedelta(days=3)
    #
    # treiZile = df[(df['Datetime'] >= start_date) & (df['Datetime'] < end_date)]
#
#     print(treiZile)
#
#     # Extragem semnalul (numărul de vehicule)
#     x = treiZile['Count'].values
#
#     # Dimensiunile ferestrei pentru filtrul de medie alunecătoare
#     window_sizes = [5, 9, 13, 17]
#
#     # Aplicam filtrul de medie alunecătoare cu diferite dimensiuni ale ferestrei
#     smooth_signals = {}
#     for w in window_sizes:
#         smoothed_signal = numpy.convolve(x, numpy.ones(w), 'valid') / w
#         smooth_signals[w] = smoothed_signal
#
#     # Afișați semnalele netezite pentru diferite dimensiuni ale ferestrei
#     for w, semnal in smooth_signals.items():
#         print(f"Window dim {w}:")
#         print(semnal)



# #c,d,e,f
# if __name__ == '__main__':
#     df = pd.read_csv('C:/Users/Vlad/Downloads/Train.csv')
#
#     df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y %H:%M')
#
#     start_date = df['Datetime'].min()
#     end_date = start_date + pd.Timedelta(days=3)
#
#     treiZile = df[(df['Datetime'] >= start_date) & (df['Datetime'] < end_date)]
#
#     ordin = 1  # ordin mai mare rezulta intr-un "slope" mai brusc
#     cutoff_frequency = 0.1  # fractiune din frecventa Nyquist, adica 0.1 din frecventa Nyquist, care este de 0.5 esantioane pe ora
#
#     #b - coef coef polinomului din numarator, iar a pt cei din numitor
#     b, a = butter(ordin, cutoff_frequency, btype='low', analog=False)
#     treiZile['Filtered_Count'] = filtfilt(b, a, treiZile['Count'])
#
#     rp = 0.1  # Nivelul de undulație în banda trecută în dB
#     b_cheby, a_cheby = cheby1(ordin, rp, cutoff_frequency, btype='low', analog=False)
#     treiZile['Filtered_Count_Cheby'] = filtfilt(b_cheby, a_cheby, treiZile['Count'])
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(treiZile['Datetime'], treiZile['Count'], label='Original')
#     plt.plot(treiZile['Datetime'], treiZile['Filtered_Count'], label='Filtered', color='red')
#     plt.plot(treiZile['Datetime'], treiZile['Filtered_Count_Cheby'], label='Chebyshev Filtered', color='green')
#     plt.xlabel('Time')
#     plt.ylabel('Count')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()
#     #Filtrul Chebyshev pare mai exact. Filtru Butterworth este mai granular atunci cand ordinul este mai mic; filtrul Chebyshev este mai granular atunci cand rp este mai mic


#Lab7

#1
# if __name__ == '__main__':
#     N = 64
#     n1, n2 = np.meshgrid(np.arange(N), np.arange(N))
#
#     x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
#     x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
#
#     Y = np.zeros((N, N), dtype=complex)
#
#     Y[0, 5] = Y[0, N - 5] = 1
#     Y[5, 0] = Y[N - 5, 0] = 1
#     Y[5, 5] = Y[N - 5, N - 5] = 1
#
#     y = ifft2(ifftshift(Y))
#
#     Y1 = fftshift(fft2(x1))
#     Y2 = fftshift(fft2(x2))
#
#     fig, axs = plt.subplots(2, 3, figsize=(12, 8))
#
#     axs[0, 0].imshow(x1, cmap='viridis', extent=(0, N, 0, N))
#     axs[0, 0].set_title('x1(n1, n2)')
#
#     axs[0, 1].imshow(np.abs(Y1), cmap='viridis', extent=(0, N, 0, N))
#     axs[0, 1].set_title('Spectru x1')
#
#     axs[0, 2].imshow(x2, cmap='viridis', extent=(0, N, 0, N))
#     axs[0, 2].set_title('x2(n1, n2)')
#
#     axs[1, 0].imshow(np.abs(Y2), cmap='viridis', extent=(0, N, 0, N))
#     axs[1, 0].set_title('Spectru x2')
#
#     axs[1, 1].imshow(np.abs(y), cmap='viridis', extent=(0, N, 0, N))
#     axs[1, 1].set_title('Poza y')
#
#     axs[1, 2].imshow(np.abs(Y), cmap='viridis', extent=(0, N, 0, N))
#     axs[1, 2].set_title('Spectru Y')
#
#     axs[1, 2].axis('off')
#
#     plt.tight_layout()
#
#     plt.savefig('C:/Users/Vlad/Downloads/spectra_and_functions.png')
#
#     # Show the figure
#     plt.show()

#2

def compress_image(img, snr_threshold):
    gray = img

    f = fft2(gray)
    fshift = fftshift(f)

    rows, cols = gray.shape

    #centru
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - int(snr_threshold):crow + int(snr_threshold), ccol - int(snr_threshold):ccol + int(snr_threshold)] = 1

    fshift = fshift * mask
    f_ishift = ifftshift(fshift)

    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


# if __name__ == '__main__':
#     X = misc.face(gray=True)
#
#     snr = 50
#
#     imagine_compresata = compress_image(X, snr)
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(X, cmap=plt.cm.gray)
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(imagine_compresata, cmap=plt.cm.gray)
#     plt.show()

def calculate_snr(signal, noise):
    mean_signal = np.mean(signal)
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)
    return 20 * np.log10(mean_signal / std_noise)

#3
# if __name__ == '__main__':
#     pixel_noise = 200
#
#     X = misc.face(gray=True)
#
#     noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=X.shape)
#     X_noisy = X + noise
#
#     # Calculul SNR înainte de eliminarea zgomotului
#     snr_before = calculate_snr(X, X_noisy - X)
#
#     # Estimarea sigma zgomotului pentru Non-Local Means Denoising
#     sigma_est = np.mean(estimate_sigma(X_noisy))
#     # Aplicarea Non-Local Means Denoising
#     X_denoised_nl_means = denoise_nl_means(X_noisy, h=1.15 * sigma_est, fast_mode=True,
#                                            patch_size=5, patch_distance=6)
#
#     # Calculul SNR după eliminarea zgomotului cu Non-Local Means Denoising
#     snr_after_nl_means = calculate_snr(X, X_denoised_nl_means - X)
#
#     # Afișarea imaginilor
#     plt.figure(figsize=(20, 7))
#     plt.subplot(1, 3, 1)
#     plt.imshow(X, cmap='gray')
#     plt.title('Original Image')
#     plt.subplot(1, 3, 2)
#     plt.imshow(X_noisy, cmap='gray')
#     plt.title('Noisy Image')
#     plt.subplot(1, 3, 3)
#     plt.imshow(X_denoised_nl_means, cmap='gray')
#     plt.title('NL Means Denoised Image')
#     plt.show()
#
#     print("SNR before noise reduction:", snr_before)
#     print("SNR after NL Means noise reduction:", snr_after_nl_means)


#Lab 8

def autocorrelation(time_series):
    # Calculează vectorul de autocorelație pentru o serie de timp dată.
    n = len(time_series)
    mean = np.mean(time_series)
    autocorr = np.correlate(time_series - mean, time_series - mean, mode='full') / np.var(time_series)
    return autocorr[n-1:] / n

#1 a b c d
# if __name__ == '__main__':
#     # Setează seed-ul pentru reproducibilitate
#     np.random.seed(0)
#
#     # Setează parametrii
#     N = 1000  # dimensiunea seriei de timp
#     t = np.arange(N)  # indexul timpului
#
#     # Generează componentele
#     # Componenta de trend (ecuație de gradul 2)
#     a, b, c = 0.001, 0.1, 5  # coeficienții pentru ecuația pătratică
#     trend = a * t ** 2 + b * t + c
#
#     # Componenta sezonieră (folosind două frecvențe)
#     freq1, freq2 = 1 / 50, 1 / 100  # frecvențele
#     season = np.sin(2 * np.pi * freq1 * t) + np.cos(2 * np.pi * freq2 * t)
#
#     # Zgomot alb gaussian
#     noise = np.random.normal(0, 1, N)
#
#     # Combina componentele pentru a forma seria de timp
#     time_series = trend + season + noise
#
#     # Desenează seria de timp și componentele sale
#     plt.figure(figsize=(14, 8))
#
#     plt.subplot(4, 1, 1)
#     plt.plot(t, time_series, label='Serie de Timp')
#     plt.title('Serie de Timp')
#     plt.legend()
#
#     plt.subplot(4, 1, 2)
#     plt.plot(t, trend, label='Trend', color='orange')
#     plt.title('Componenta de Trend')
#     plt.legend()
#
#     plt.subplot(4, 1, 3)
#     plt.plot(t, season, label='Sezonieră', color='green')
#     plt.title('Componenta Sezonieră')
#     plt.legend()
#
#     plt.subplot(4, 1, 4)
#     plt.plot(t, noise, label='Zgomot', color='red')
#     plt.title('Zgomot (Zgomot Alb Gaussian)')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
#     autocorr_vector = autocorrelation(time_series)
#
#     # Desenăm vectorul de autocorelație
#     plt.figure(figsize=(10, 5))
#     plt.stem(autocorr_vector)
#     plt.title('Vectorul de Autocorelație al Seriei de Timp')
#     plt.xlabel('Lag')
#     plt.ylabel('Autocorelație')
#     plt.show()
#
#     p = 5
#
#     # Crearea și antrenarea modelului AR
#     model_ar = AutoReg(time_series, lags=p)
#     model_ar_fit = model_ar.fit()
#
#     # Obținerea predicțiilor
#     predictions = model_ar_fit.predict(start=p, end=N - 1, dynamic=False)
#
#     # Desenăm seria de timp originală și predicțiile
#     plt.figure(figsize=(14, 6))
#     plt.plot(time_series, label='Serie de Timp Originală', color='blue')
#     plt.plot(range(p, N), predictions, label='Predicții AR', color='red', linestyle='dashed')
#     plt.title('Serie de Timp și Predicțiile Modelului AR')
#     plt.legend()
#     plt.show()
#
#
#
#
#
#     # Împarte datele în seturi de antrenament și testare
#     train_size = int(N * 0.8)
#     train, test = time_series[:train_size], time_series[train_size:]
#
#     # Căutarea în grilă pentru cei mai buni parametri p și m (întârzieri și orizont de predicție)
#     p_values = range(1, 11)  # interval pentru p
#     m_values = range(1, 6)  # interval pentru m
#     best_score, best_cfg = float("inf"), None
#
#     for p in p_values:
#         for m in m_values:
#             # Antrenează modelul AR
#             model = AutoReg(train, lags=p)
#             model_fit = model.fit()
#
#             # Face predicții
#             predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
#
#             # Folosește doar primele m predicții
#             predictions = predictions[:m]
#
#             # Calculează eroarea
#             error = mean_squared_error(test[:m], predictions)
#             if error < best_score:
#                 best_score, best_cfg = error, (p, m)
#             print(f'AR({p}) Predict {m} pasi: MSE={error:.3f}')
#
#     print(f'Cea mai bună configurație AR(p) Predict(m) pasi: {best_cfg} cu MSE={best_score:.3f}')

#Lab9
#Ex1,2,3
if __name__ == '__main__':
    # Seteaza seed-ul pentru reproducibilitate
    np.random.seed(0)

    # Seteaza parametrii
    N = 1000  # Dimensiunea seriei de timp
    t = np.arange(N)  # Indexul timpului

    # Genereaza componentele
    # Componenta de trend (ecuatie de gradul 2)
    a, b, c = 0.001, 0.1, 5  # Coeficientii pentru ecuatia patratica
    trend = a * t ** 2 + b * t + c

    # Componenta sezoniera (folosind doua frecvente)
    freq1, freq2 = 1 / 50, 1 / 100  # Frecventele
    season = np.sin(2 * np.pi * freq1 * t) + np.cos(2 * np.pi * freq2 * t)

    # Zgomot alb gaussian
    noise = np.random.normal(0, 1, N)

    # Combina componentele pentru a forma seria de timp
    time_series = trend + season + noise


    # Functia pentru Medierea Exponentiala (EMA)
    def exponential_moving_average(data, alpha):
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema


    # Calculeaza EMA cu un alpha fix
    alpha_fixed = 0.1
    ema_fixed = exponential_moving_average(time_series, alpha_fixed)


    # Gaseste un alpha optim
    # Pentru simplitate, folosim eroarea patratica medie (MSE) fata de componenta de trend
    # ca masura a cat de bine EMA capteaza trendul subiacent.
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    alphas = np.linspace(0, 1, 100)
    optimal_alpha = alphas[0]
    lowest_mse = mse(trend, exponential_moving_average(time_series, alphas[0]))

    for alpha in alphas[1:]:
        current_ema = exponential_moving_average(time_series, alpha)
        current_mse = mse(trend, current_ema)
        if current_mse < lowest_mse:
            lowest_mse = current_mse
            optimal_alpha = alpha

    ema_optimal = exponential_moving_average(time_series, optimal_alpha)
    plt.figure(figsize=(14, 8))

    # Plotează seria de timp
    plt.plot(t, time_series, label='Serie de Timp', color='black', linewidth=2)

    # Plotează EMA fixat și EMA optimizat cu linii punctate
    plt.plot(t, ema_fixed, label=f'EMA Fixat (α = 0.1)', color='purple', linestyle='dotted')
    plt.plot(t, ema_optimal, label=f'EMA Optim (α = {optimal_alpha:.2f})', color='yellow', linestyle='dotted')

    plt.title('Comparare între Seria de Timp, EMA Fixat și EMA Optimizat')
    plt.legend()

    plt.show()



    # Setam orizontul q pentru modelul Moving Average (MA)
    q = 5

    # Cream modelul MA folosind seria de timp si orizontul q specificat
    # Pentru un model MA pur, setam p (AR) = 0 si d (diferentiere) = 0
    model_ma = ARIMA(time_series, order=(0, 0, q))
    model_ma_fit = model_ma.fit()

    # Obtinem predictiile modelului MA
    ma_predictions = model_ma_fit.predict(start=0, end=N - 1)

    # Ploteaza seria de timp si predictiile modelului MA
    plt.figure(figsize=(14, 8))
    plt.plot(time_series, label='Serie de Timp Originala', color='black')
    plt.plot(ma_predictions, label=f'Model MA (q={q})', color='red', linestyle='dashed')
    plt.title(f'Seria de Timp si Predictiile Modelului MA (q={q})')
    plt.legend()
    plt.show()















