from typing import Union, Optional
import numpy as np

"""
Helper Funktionen
"""


def tridiag(rows: int, cols: int, lower: float, center: float, upper: float) -> np.array:
    """
    Erstellt Tri-Diagonal-Matrix der dimension [rows x cols]

    Examples:
        >>> tridiag(5,5,1,2,3)
        array([[2., 3., 0., 0., 0.],
               [1., 2., 3., 0., 0.],
               [0., 1., 2., 3., 0.],
               [0., 0., 1., 2., 3.],
               [0., 0., 0., 1., 2.]])


    :param rows: Anzahl Reihen
    :param cols: Anzahl Spalten
    :param lower: linkes/unteres Diagonalelement
    :param center: centrales Diagonalelement
    :param upper: rechtes/oberes Diagonalelement
    :return: Tri-Diagonal-Matrix
    """
    out = np.zeros([rows, cols], dtype=float)
    for row in range(rows):
        if row > 0:
            out[row][row - 1] = lower
        out[row][row] = center
        if row < rows - 1:
            out[row][row + 1] = upper
    return out


def vec_size(vector: np.array) -> int:
    """
    Helper Funktion für Vektordimension

    :param vector: Zeilen oder Spaltenvektor
    :returns: Dimension des Vektors
    """
    if len(vector.shape) > 1:
        return vector.shape[0] if vector.shape[0] > vector.shape[1] else vector.shape[1]
    return vector.size


"""
Exceptions
"""


class ExtendedException(Exception):
    """
    Helper Klasse für vereinfachte Exceptions
    """
    message = ''

    def __init__(self, extra: str = None):
        if extra:
            self.message += f' {extra}'
        super().__init__(self.message)


class WrongDimensionError(ExtendedException):
    """
    Wird geworfen wenn falsche Dimensionen übergeben werden
    """
    message = 'Falsche Dimensionen!'


class NotEquidistantSteps(ExtendedException):
    """
    Wird geworfen wenn Stützwerte mit nicht äuqidistanter Distanz
    in x_{j+1}-x_j für j = 0, ..., n-1 übergeben wurde.
    """
    message = 'Erwartete äquidistante Schrittweite!'

    def __init__(self, h, expected_h):
        super().__init__(f'{h} != {expected_h}')


"""
Tests für Eingabeparameter
"""


def parameter_test_decorator(min_elements: int = 2, test_size: bool = True, test_equidistant: bool = False):
    """
    Decorator Implementation zur Verallgemeinerung von
    Parametertests zu Beginn eines Funktionsaufrufs von polinterpol und splineinterpol.
    :param min_elements: minimale Anzahl an übergebenen Stützwerten
    :param test_size:    Test nach gleicher Anzahl Werte in xi und yi
    :param test_equidistant: Teste auf equidistante Schrittweite in xi
    """

    def wrapper(fun):
        def inner(*args, **kwargs):
            x = kwargs.pop('x', None)
            if x is None:
                x = args[0]
            xi = kwargs.pop('xi', None)
            if xi is None:
                xi = args[1]
            yi = kwargs.pop('yi', None)
            if yi is None:
                yi = args[2]

            xi_size = vec_size(xi)
            yi_size = vec_size(yi)

            # Test auf mindest Anzahl Stützwerte
            if xi_size < min_elements:
                raise WrongDimensionError(f'Zu wenig Stützwerte xi({len(xi)}) übergeben.')
            if vec_size(x) < 1:
                raise WrongDimensionError(
                    f'Zu wenig Werte zur Approximation übergeben, benötigt mindestens einen Wert!')

            # Test auf gleiche Anzahl Elemente in xi und yi
            if test_size:
                if xi_size != yi_size:
                    raise WrongDimensionError(f'xi({xi_size}) != yi({yi_size})')

            h = xi[1] - xi[0]

            # Test auf äquidistante Schrittweite innerhalb xi
            if test_equidistant:
                for i in range(xi_size - 1):
                    if abs((xi[i + 1] - xi[i]) - h) > 1e-10:
                        raise NotEquidistantSteps(xi[i + 1] - xi[i], h)

            return fun(x=x, xi=xi, yi=yi)

        return inner

    return wrapper


@parameter_test_decorator()
def polinterpol(x: np.array, xi: np.array, yi: np.array, **kwargs) -> np.array:
    """
    Bildet Interpolationspolynom in der Newton-Basis und wertet
    übergebene Werte aus.

    :param x: zu approximierende Stellen
    :param xi: Stützwerte
    :param yi: Funktionswerte der Stützstellen
    :return: Interpolierte Funktionswerte
    """

    def newton_base_coefficients() -> np.array:
        """
        Berechnung mit dem rekursiven Schema der dividierten Differenzen - iterativ implementiert
        :return: Newton-Basiskoeffizienten für gegebene Stützwerte
        """
        m = len(xi)
        f = np.zeros((m, m))
        f[:, 0] = yi
        for j in range(1, m):
            for i in range(j, m):
                f[i, j] = (f[i, j - 1] - f[i - 1, j - 1]) / (xi[i] - xi[i - j])
        return np.diag(f)

    def pol_eval_newton_horner(b: np.array) -> np.array:
        """ Polynomonauswertung in Newton-Basis mit Hornerschema

        :param b:       Newton-Basis-Koeffizienten des Polynoms
        :return: Ausgewertete Funktionswerte
        """
        y = np.ones(x.shape) * b[-1]
        for j in range(b.shape[0] - 2, -1, -1):
            # calculates hadamard-multiplication
            y = b[j] + (x - xi[j]) * y
        return y

    return pol_eval_newton_horner(newton_base_coefficients())


@parameter_test_decorator(test_equidistant=True)
def splineinterpol(x: Union[np.array, float], xi: np.array, yi: np.array) -> Union[np.array, float]:
    """
    Auswertung des natürlichen kubischen Splines
    :param x :      auszuwertende Stellen
    :param xi:      Stützwerte
    :param yi:      Stützstellen
    """

    # Berechnung von optionalen Parametern
    h = xi[1] - xi[0]

    xi_size = vec_size(xi)

    def curv(step_width: float) -> np.array:
        """
        Berechnung der zweiten Ableitungen an den Knoten
        :param step_width: äquidistante Schrittweite zwischen Stützwerten
        :return: Koeffizienten m_i
        """
        m_i = np.zeros(xi.shape)
        dim = xi.shape[0] - 2
        rhs = 6 / (step_width ** 2) * np.dot(tridiag(dim, dim, 1, -2, 1), yi[1:-1])
        lhs = tridiag(dim, dim, 1, 4, 1)
        m_i[1:-1] = np.linalg.solve(lhs, rhs)
        return m_i

    def eval_spline_j(x_act: Union[np.array, float], index: int, m_i: np.array) -> Union[np.array, float]:
        """
        Auswertung des kubischen Polynoms s_{index}
        :param x_act: aktueller Stützwert
        :param index: index des Splines
        :param m_i:   koeffizienten des kubisch natürlichen Splines
        :return:      Wert des Splines an gegebener Stelle
        """
        return ((m_i[index] * (xi[index + 1] - x_act) ** 3 + m_i[index + 1] * (x_act - xi[index]) ** 3.0) / (6.0 * h)
                + (yi[index] * (xi[index + 1] - x_act) + yi[index + 1] * (x_act - xi[index])) / h
                - ((m_i[index] * (xi[index + 1] - x_act) + m_i[index + 1] * (x_act - xi[index])) * h) / 6.0)

    mi: np.array = curv(h)
    y: np.array = np.zeros(x.shape)
    # Werte die splines elementweise aus, für alle
    # elemente xi für i = 0, ..., n-1
    for j in range(xi_size - 1):
        # alle elemente in x, die zwischen x_j und x_{j+1} liegen
        ind: np.array = np.nonzero((xi[j] <= x) & (x < xi[j + 1]))
        # werden mit dem j-ten spline ausgewertet
        y[ind] = eval_spline_j(x[ind], j, mi)

    return y
