import tensorflow as tf
from src.utils import load_data

# Load the model
model = tf.keras.models.load_model('models/model.h5')

# Load test data
test_data = load_data('data/test')

# Evaluate the model
loss, accuracy = model.evaluate(test_data)

# Save evaluation metrics
with open('results/evaluation_metrics.txt', 'w') as f:
    f.write(f'Loss: {loss}\n')
    f.write(f'Accuracy: {accuracy}\n')

print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
