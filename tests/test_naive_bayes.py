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
        y = np.asarray(y)
        self.nb.calculate_priors(y)
        expected_priors = {
            "1": 0.25,
            "3": 0.5,
            "4": 0.25,
        }
        self.assertEqual(self.nb.prior_probs, expected_priors)

    def test_calculate_probabilites(self):
        """Test probability calculation."""
        X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        y = [0, 1, 0]
        y = np.asarray(y)
        self.nb.calculate_priors(y)
        self.nb.calculate_feature_conditional(X, y)

    def test_predict(self):
        """Test predict calculation."""
        X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        y = [0, 1, 0]
        y = np.asarray(y)
        self.nb.fit(X, y)
        y_pred = self.nb.predict(X)
        print(y_pred)

    def test_evaluate(self):
        X = csr_matrix([[1, 0, 1], [0, 1, 1], [1, 0, 0]])
        y = [0, 1, 0]
        y = np.asarray(y)
        self.nb.fit(X, y)
        y_pred, y_pred_range = self.nb.predict(X)
        print(self.nb.evaluate_acc(y, y_pred_range))
        print(self.nb.evaluate_acc_confusion(y, y_pred))


    # def test_fit_predict_real(self):
    #     """Test fit predict pipeline on real dataset."""
    #     df = get_data()
    #     X, y = vectorize_get_X_y(df)
    #     self.nb.fit(X, y)
    #     self.nb.predict(X)




    def test_get_data(self):
        """Test data loading."""
        train, val, test = get_data()
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        self.assertIn("text", train.columns)



if __name__ == "__main__":
    unittest.main()
