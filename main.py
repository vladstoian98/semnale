import numpy
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    # ex2f
    matr = numpy.zeros((128, 128))

    timp = numpy.linspace(0, 0.03, 128)

    sinValori = (s_sin(1, 100, timp, 0) + 1) * 64

    sinValoriIndici = sinValori.astype(int)

    for i, y in enumerate(sinValoriIndici):
        if 0 <= y < 128:
            matr[y, i] = 1

    plt.imshow(matr, cmap='gray')
    plt.show()












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

if __name__ == '__main__':
    timp = numpy.linspace(0, 1, 50, endpoint=False)
    sinSemnal = sinus(1, 440, timp, 0)
    sawSemnal = sawtooth(2 * numpy.pi * 440 * timp)

    semnalTotal = sinSemnal + sawSemnal

    print(numpy.sum(semnalTotal))

    fig, axs = plt.subplots(3)
    axs[0].plot(timp, sinSemnal)
    axs[1].plot(timp, sawSemnal)
    axs[2].plot(timp, semnalTotal)

    plt.show()




















