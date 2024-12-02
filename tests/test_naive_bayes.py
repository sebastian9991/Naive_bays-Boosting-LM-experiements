import unittest
from naive import NaiveBayes, get_data, vectorize_get_X_y
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


class TestNaiveBayes(unittest.TestCase):
    def setUp(self):
        """Set up test data and objects."""
        self.nb = NaiveBayes()
        # Mock data
        self.mock_data = pd.DataFrame({
            "text": ["happy day", "sad moment", "angry reaction", "neutral"],
            "y": [1, 3, 3, 4],
        })


    def test_calculate_priors(self):
        """Test prior probability calculation."""
        y = self.mock_data.drop(columns=["text"])['y']
        print(y)
        y = np.asarray(y)
        self.nb.fit(None, y)
        expected_priors = {
            "1": 0.25,
            "3": 0.5,
            "4": 0.25,
        }
        self.assertEqual(self.nb.prior_probs, expected_priors)

    def test_calculate_probabilites(self):
        """Test probability calculation."""
        X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        y = ['class1', 'class2', 'class1']
        y = np.asarray(y)
        self.nb.calculate_priors(y)
        self.nb.calculate_feature_conditional(X, y)




    # def test_vectorize_get_X_y(self):
    #     """Test the vectorization and splitting."""
    #     X, y = vectorize_get_X_y(self.mock_data)
    #     self.assertIsInstance(X, csr_matrix)
    #     self.assertEqual(X.shape[0], len(self.mock_data))
    #     self.assertEqual(y.shape[0], len(self.mock_data))
    #     self.assertIn("happy", y.columns)

    # def test_fit_and_predict(self):
    #     """Test the fit and predict methods."""
    #     # Mock data
    #     X, y = vectorize_get_X_y(self.mock_data)
    #     y = y.idxmax(axis=1)  # Simplify to single-label for testing
    #     self.nb.fit(X, y)
    #     predictions = self.nb.predict(X)
    #     self.assertEqual(len(predictions), len(y))
    #     self.assertTrue(all(pred in y.unique() for pred in predictions))


    def test_get_data(self):
        """Test data loading."""
        df = get_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("text", df.columns)



if __name__ == "__main__":
    unittest.main()
