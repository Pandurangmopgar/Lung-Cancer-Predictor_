import requests
import unittest

class TestLungCancerPrediction(unittest.TestCase):

    def test_predict_endpoint(self):
        url = "http://127.0.0.1:5000/predict"  # Replace with your Flask app's URL

        # Test with a valid image
        with open("E:/Lung_Cancer_CNN/Data/Data/Dataset/test/normal/11 - Copy (2).png", "rb") as f:
            response = requests.post(url, files={"image": f})
            self.assertIn(response.json()["prediction"], ['adinocarcinoma', 'carcinoma', 'normal', 'squamous'])

        # Add more test cases here, such as invalid image formats, etc.

if __name__ == "__main__":
    unittest.main()
