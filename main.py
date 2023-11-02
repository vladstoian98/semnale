import numpy
import matplotlib.pyplot as plt
import time
import sounddevice as sd
from scipy.signal import sawtooth

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





#Lab 4
if __name__ == '__main__':
    #Ex1
    N = [128, 256, 512, 1024, 2048, 4096, 8192]
    timpiTrecutiProprii = []
    timpiTrecutiPython = []

    for i in N:
        t = numpy.linspace(0, 1, i, endpoint=False)
        f1, f2, f3 = 20, 30, 100
        semnaleConcatenate = numpy.sin(2 * numpy.pi * f1 * t) + numpy.sin(2 * numpy.pi * f2 * t) + numpy.sin(2 * numpy.pi * f3 * t)

        fftPropriuT1 = time.perf_counter()

        vect = numpy.zeros(i, complex)

        for k in range(i):
            for n in range(i):
                vect[k] += semnaleConcatenate[n] * numpy.exp(-2j * numpy.pi * k * n / i)

        fftPropriuT2 = time.perf_counter()
        timpPropriu = fftPropriuT2 - fftPropriuT1
        timpiTrecutiProprii.append(timpPropriu)


    for i in N:
        t = numpy.linspace(0, 1, i, endpoint=False)
        f1, f2, f3 = 20, 30, 100
        semnaleConcatenate = numpy.sin(2 * numpy.pi * f1 * t) + numpy.sin(2 * numpy.pi * f2 * t) + numpy.sin(2 * numpy.pi * f3 * t)

        fftPythonT1 = time.perf_counter()

        fftVar = numpy.fft.fft(semnaleConcatenate)

        fftPythonT2 = time.perf_counter()

        timpPython = fftPythonT2 - fftPythonT1
        timpiTrecutiPython.append(timpPython)

    timpiTrecutiPropriiMs = [x * 1000 for x in timpiTrecutiProprii]
    timpiTrecutiPythonMs = [x * 1000 for x in timpiTrecutiPython]

    print(timpiTrecutiPython)

    plt.figure(figsize=(10, 5))
    plt.plot(N, timpiTrecutiPropriiMs, 'o-', color='red')
    plt.plot(N, timpiTrecutiPythonMs, 'o-', color='green')
    plt.yscale('log')
    plt.xlabel('Size of N')
    plt.ylabel('Elapsed Time (milliseconds)')
    plt.grid(True, which="both", ls="--")

    plt.show()





























