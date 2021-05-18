import numpy as np
from interpol1d import polinterpol, tschebyscheff, splineinterpol
import matplotlib.pyplot as plt


# Teilaufgabe a)
def eval_runge(n: int = 3):
    def runge(_x):
        """runge-funktion: https://en.wikipedia.org/wiki/Runge%27s_phenomenon"""
        return 1 / (1 + _x ** 2)
    # Parameterwahl
    realX = np.linspace(-5, 5, n) # aequidistante Stuetzstellen im Intervall [-5, 5]
    xi = tschebyscheff(-5, 5, realX) # wahl der tschebyscheff stützpunkte
    yi = runge(xi)  # auswertung mit Funktion für Stuezstellen

    x = np.linspace(np.min(xi), np.max(xi), 100)  # zu interpolierende werte

    # Polynominterpolation mit Newton-Basis
    out_newton = polinterpol(x, xi, yi)
    # Splineinterpolation mit natürlichen Splines
    out_spline = splineinterpol(x, xi, yi)

    # Darstellung
    plt.rcParams.update({'font.size': 14})
    plt.plot(xi, yi, '*', label='Stützwerte', linewidth=2, markersize=10)
    plt.plot(x, out_newton, label='Newton-Basis', linewidth=2)
    plt.plot(x, out_spline, label='Natürliche Splines', linewidth=2)
    plt.legend(loc='upper right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Rungefunktion $f(x) = 1/(1+x^2)$ mit {len(realX)} Stützstellen')
    plt.show()



def eval_big_O(n: int = 3):
    def runge(_x):
        """runge-funktion: https://en.wikipedia.org/wiki/Runge%27s_phenomenon"""
        return 1 / (1 + _x ** 2)
    # Parameterwahl
    realX = np.linspace(-5, 5, n) # aequidistante Stuetzstellen im Intervall [-5, 5]
    xi = tschebyscheff(-5, 5, realX) # wahl der tschebyscheff stützpunkte
    yi = runge(xi)  # auswertung mit Funktion für Stuezstellen

    x = np.linspace(np.min(xi), np.max(xi), 100)  # zu interpolierende werte

    # Splineinterpolation mit natürlichen Splines
    out_spline = splineinterpol(x, xi, yi)
    
    err[int(n/3)-1] = np.linalg.norm(out_spline-runge(x), np.inf)



if __name__ == '__main__':

    # 2a)
    
    for i in range(3, 21, 3):
        eval_runge(i)

    # 2b)

    n = 150

    err = np.zeros(n)
    
    for i in range(len(err-2)):
        eval_big_O(i*3+3)


    # Darstellung
    plt.rcParams.update({'font.size': 14})
    plt.semilogy(err)
    x = np.flip(np.linspace(3, n, 30))
    #plt.xlim(3, n)
    #plt.legend(loc='upper right')
    plt.xlabel('# Stützstellen')
    plt.ylabel('Fehler')
    plt.show()
    