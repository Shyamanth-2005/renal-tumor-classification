import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

# Set up logging
logger = logging.getLogger(__name__)


class PredictionPipeline:
  def __init__(self, filename):
    self.filename = filename

  def predict(self):
    try:
      # Check if file exists
      if not os.path.exists(self.filename):
        logger.error(f"Image file does not exist: {self.filename}")
        raise FileNotFoundError(f"Image file does not exist: {self.filename}")

      # Check file size
      file_size = os.path.getsize(self.filename)
      if file_size == 0:
        logger.error(f"Image file is empty: {self.filename}")
        raise ValueError(f"Image file is empty: {self.filename}")

      # Validate file extension
      valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
      _, ext = os.path.splitext(self.filename.lower())
      if ext not in valid_extensions:
        logger.error(f"Invalid image file format: {ext}. Supported formats: {valid_extensions}")
        raise ValueError(f"Invalid image file format: {ext}. Supported formats: {valid_extensions}")

      # Load model with custom objects to handle potential compatibility issues
      model_path = os.path.join("artifacts", "training", "model.h5")
      if not os.path.exists(model_path):
        logger.error(f"Model file does not exist: {model_path}")
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

      # Load model with compatibility adjustments for reduction parameter issue
      import tensorflow as tf

      # Load the model with compile=False to avoid issues with loss function
      model = load_model(model_path, compile=False)

      # Recompile the model to handle potential compatibility issues
      model.compile(
          optimizer='adam',  # Use a standard optimizer
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy']
      )

      # Load and preprocess image
      imagename = self.filename
      test_image = image.load_img(imagename, target_size=(224, 224))
      test_image = image.img_to_array(test_image)
      test_image = np.expand_dims(test_image, axis=0)

      # Normalize the image if the model expects normalized input (values between 0 and 1)
      test_image = test_image / 255.0

      # Make prediction
      predictions = model.predict(test_image)
      print(f"Raw predictions: {predictions}")

      # Get the predicted class and confidence
      result = np.argmax(predictions, axis=1)
      confidence = np.max(predictions)  # Get confidence score

      # Also get the probability for each class
      probabilities = predictions[0]  # Get probabilities for the single image
      print(f"Class probabilities: Normal={probabilities[0]:.4f}, Tumor={probabilities[1]:.4f}")
      print(f"Prediction result: {result}, Confidence: {confidence}")

      # Determine prediction - assuming index 0 is 'Normal' and index 1 is 'Tumor'
      if result[0] == 1:  # Tumor class
        prediction = 'Tumor'
        confidence_percentage = round(confidence * 100, 2)
        return [{"image": prediction, "confidence": f"{confidence_percentage}%", "details": {"normal_prob": f"{probabilities[0]:.4f}", "tumor_prob": f"{probabilities[1]:.4f}"} }]
      else:  # Normal class
        prediction = 'Normal'
        confidence_percentage = round(confidence * 100, 2)
        return [{"image": prediction, "confidence": f"{confidence_percentage}%", "details": {"normal_prob": f"{probabilities[0]:.4f}", "tumor_prob": f"{probabilities[1]:.4f}"} }]

    except FileNotFoundError as e:
      logger.error(f"File not found error: {str(e)}")
      return [{"error": f"File not found: {str(e)}", "image": "Error"}]

    except ValueError as e:
      logger.error(f"Value error: {str(e)}")
      return [{"error": f"Invalid input: {str(e)}", "image": "Error"}]

    except Exception as e:
      logger.error(f"Unexpected error during prediction: {str(e)}")
      return [{"error": f"Prediction failed: {str(e)}", "image": "Error"}]