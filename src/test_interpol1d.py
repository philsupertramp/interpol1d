import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.interpol1d import polinterpol, WrongDimensionError, tridiag, vec_size, splineinterpol, NotEquidistantSteps


class PolInterpolTestCase(unittest.TestCase):
    def test_correctness(self):
        xi = np.array([0, 1 / 24, 1 / 12, 1 / 6], dtype=float)  # aequidistante Stuetzstellen
        yi = 2 * np.sin(3 * np.pi * xi)
        x = np.linspace(0, 1 / 6.0, 10)

        out1 = polinterpol(x, xi, yi)
        self.assertEqual(out1.all(), np.array([0., 0.34874939, 0.68442336, 0.9990692, 1.28473415, 1.53346548,
                                               1.73731047, 1.88831638, 1.97853046, 2.]).all())
        self.assertEqual(out1.all(), polinterpol(x=xi, xi=xi, yi=yi).all())

    def test_exception_for_wrong_dimension_xi_yi(self):
        xi = np.array([1, 2])
        yi = np.array([1])
        x = np.linspace(0, 1, 3)
        with self.assertRaises(WrongDimensionError) as ex:
            polinterpol(x, xi, yi)
            self.assertEqual(str(ex), f'Falsche Dimensionen! xi(2) != yi(1)')

    def test_exception_for_min_size_xi(self):
        xi = np.array([1])
        yi = np.array([1, 2])
        x = np.array([1, 2])
        with self.assertRaises(WrongDimensionError) as ex:
            polinterpol(x, xi, yi)
            self.assertEqual(str(ex), f'Falsche Dimensionen! Zu wenig Stützwerte xi(1) übergeben.')

        xi = np.array([])
        with self.assertRaises(WrongDimensionError) as ex:
            polinterpol(x, xi, yi)
            self.assertEqual(str(ex), f'Falsche Dimensionen! Zu wenig Stützwerte xi(0) übergeben.')

    def test_exception_for_min_size_x(self):
        yi = np.array([1, 2])
        xi = np.array([1, 2])
        x = np.array([])
        with self.assertRaises(WrongDimensionError) as ex:
            polinterpol(x, xi, yi)
            self.assertEqual(str(ex), 'Falsche Dimensionen! Zu wenig Werte zur Approximation übergeben, benötigt '
                                      'mindestens einen Wert!')

    def test_ignores_equidistance(self):
        xi = np.array([0, 1 / 24, 1 / 6, 1 / 2], dtype=float)  # aequidistante Stuetzstellen
        yi = 2 * np.sin(3 * np.pi * xi)
        x = np.linspace(0, 1 / 6.0, 10)
        dists = []
        for i in range(2):
            dists.append(xi[i + 1] - xi[i])
        self.assertNotEqual(dists[0], dists[1])

        # execution runs without exception
        polinterpol(x, xi, yi)


class HelperTestCase(unittest.TestCase):
    def test_tridiag(self):
        out = tridiag(5, 5, 1, 2, 3)
        self.assertEqual(out.all(), np.array([[2., 3., 0., 0., 0.],
                                              [1., 2., 3., 0., 0.],
                                              [0., 1., 2., 3., 0.],
                                              [0., 0., 1., 2., 3.],
                                              [0., 0., 0., 1., 2.]]).all())

        out = tridiag(2, 2, 1, 2, 3)
        self.assertEqual(out.all(), np.array([[2., 3.],
                                              [1., 2.]]).all())

    def test_vec_size(self):
        x = np.array([1, 2, 3])
        self.assertEqual(vec_size(x), 3)
        x = x.reshape([1, 3])
        self.assertEqual(vec_size(x), 3)


class SplineInterpolTestCase(unittest.TestCase):
    def test_correctness(self):
        xi = np.array([0, 1 / 24, 1 / 12, 3 / 24], dtype=float)  # aequidistante Stuetzstellen
        yi = 2 * np.sin(3 * np.pi * xi)
        x = np.linspace(0, 1 / 6.0, 10)
        expected = np.array([0., 0.30219189, 0.66046426, 1.1029231, 1.41419245, 1.67721466, 1.86394949, 0., 0., 0.])

        out = splineinterpol(x=x, xi=xi, yi=yi)

        self.assertEqual(out.all(), expected.all())

    def test_exception_for_size_xi(self):
        xi = np.array([1])
        yi = np.array([1, 2])
        x = np.array([1, 2])
        with self.assertRaises(WrongDimensionError) as ex:
            splineinterpol(x, xi, yi)
            self.assertEqual(str(ex), f'Falsche Dimensionen! Zu wenig Stützwerte xi(1) übergeben.')

        xi = np.array([])
        with self.assertRaises(WrongDimensionError) as ex:
            splineinterpol(x, xi, yi)
            self.assertEqual(str(ex), f'Falsche Dimensionen! Zu wenig Stützwerte xi(0) übergeben.')

    def test_exception_for_size_x(self):
        yi = np.array([1, 2])
        xi = np.array([1, 2])
        x = np.array([])
        with self.assertRaises(WrongDimensionError) as ex:
            splineinterpol(x, xi, yi)
            self.assertEqual(str(ex), 'Falsche Dimensionen! Zu wenig Werte zur Approximation übergeben, benötigt '
                                      'mindestens einen Wert!')

    def test_exception_for_step_width(self):

        xi = np.array([0, 1/2, 1/4], dtype=float)  # aequidistante Stuetzstellen
        yi = 2 * np.sin(3 * np.pi * xi)
        x = np.linspace(0, 1 / 6.0, 10)

        with self.assertRaises(NotEquidistantSteps) as ex:
            splineinterpol(x, xi, yi)
            self.assertEqual(str(ex), 'Erwartete äquidistante Schrittweite! 0.25 != 0.5')


if __name__ == '__main__':
    unittest.main()
