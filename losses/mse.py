import unittest
import numpy as np

class MSE:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        returns mean squared error of all y_true[i]-y_pred[i]
        """

        #assert y_true and y_pred are same shape
        if y_true.shape != y_pred.shape:
            raise ValueError("Expected two np.ndarrays of same shape but got two of different shapes")
        else:
            return np.mean((y_true - y_pred)**2)



class TestMSE(unittest.TestCase):
    def test_compute_mse(self):
        mse = MSE()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        expected_mse = ((0)**2 + (0)**2 + (1)**2) / 3
        self.assertAlmostEqual(mse.compute(y_true, y_pred), expected_mse)

    def test_shape_mismatch(self):
        mse = MSE()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        with self.assertRaises(ValueError):
            mse.compute(y_true, y_pred)

if __name__ == "__main__":
    unittest.main()
