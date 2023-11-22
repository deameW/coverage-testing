import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

class ModelTrainer:
    def __init__(self, model_name, input_shape, num_classes, dataset_name):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.dataset_name = dataset_name

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(6, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu'))
        model.add(Dense(84, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
        # Data preprocessing
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Ensure the labels are one-hot encoded
        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        # Set model save path
        model_save_dir = f"../models/{self.model_name}"
        model_save_path = os.path.join(model_save_dir, f"{self.model_name}.h5")

        # Define a callback to save the best weights
        checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        # Train the model
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[checkpoint])

        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print("Test Accuracy:", test_acc)

# 使用示例

if __name__ == '__main__':
    trainer = ModelTrainer(model_name="lenet5", input_shape=(28, 28, 1), num_classes=10)
    # trainer.train_model(x_train, y_train, x_test, y_test, epochs=10, batch_size=64)