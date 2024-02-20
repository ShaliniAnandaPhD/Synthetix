import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

def main():
    model = create_model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Load your dataset here
    # model.fit(X_train, Y_train, epochs=10)
    model.save('models/depth_model.h5')

if __name__ == "__main__":
    main()
