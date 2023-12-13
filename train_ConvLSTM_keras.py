from load_dataset import EventsToWords
from conv_lstm_model import ConvLSTM
import numpy as np
import os

# Constants
DATASET_PATH = "smemi309-final-evaluation-challenge-2022"
TIME_BINS = 8
RESIZE_SCALE = 0.6
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10

def main():
    # Load the dataset
    dataset = EventsToWords(directory=DATASET_PATH, time_bins=TIME_BINS, resize_scale=RESIZE_SCALE)
    X_train, y_train, y_train_labels = dataset.get_training_data()

    # Initialize and train the model
    model = ConvLSTM(X_train, y_train)
    history = model.train(epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE)

if __name__ == "__main__":
    main()
