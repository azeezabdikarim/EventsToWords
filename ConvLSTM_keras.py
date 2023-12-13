from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

class convLSTM():
    def __init__(self, X_train, y_train, conv_filters = [4,4], dropout = 0.4):
        self.X_train = X_train
        self.y_train = y_train
        self.IMAGE_HEIGHT = X_train[0][0].shape[0]
        self.IMAGE_WIDTH = X_train[0][0].shape[1]
        self.SEQUENCE_LENGTH = len(X_train[0])
        self.OUTPUT_DIM = 10
        self.conv_filter_depths = conv_filters
        self.dropout = dropout
        self.model = self.create_convlstm_model()
    
    def train(self, epochs = 10, batch_size = 32, patience = 10):
        early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = patience, mode = 'min', restore_best_weights = True)
        self.model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ["accuracy"])
        convlstm_model_training_history = self.model.fit(x = self.X_train, y = self.y_train, epochs = epochs, 
                                                         batch_size = batch_size, shuffle = True, validation_split = 0.2, 
                                                         callbacks = [early_stopping_callback])
        return convlstm_model_training_history
        
    def create_convlstm_model(self):
        # We will use a Sequential model for model construction
        model = Sequential()

        # Define the Model Architecture.
        first_filter_depth, second_filter_depth = self.conv_filter_depths[0], self.conv_filter_depths[1]
        
        model.add(ConvLSTM2D(filters = first_filter_depth, kernel_size = (3, 3), activation = 'tanh',data_format = "channels_last",
                             recurrent_dropout=self.dropout, return_sequences=True, input_shape = (self.SEQUENCE_LENGTH,
                                                                                          self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1)))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(TimeDistributed(Dropout(0.4)))
        model.add(ConvLSTM2D(filters = second_filter_depth, kernel_size = (3, 3), activation = 'tanh', data_format = "channels_last",
                             recurrent_dropout=self.dropout, return_sequences=True))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
        model.add(Flatten()) 
        model.add(Dense(self.OUTPUT_DIM, activation = "softmax"))

        # Return the constructed convlstm model.
        return model

# Example usage:
# model = ConvLSTM(X_train, y_train)
# convlstm_model = model.create_convlstm_model()
# convlstm_model.summary()