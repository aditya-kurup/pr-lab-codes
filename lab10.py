import tensorflow as tf from tensorflow.keras import datasets, layers, models import matplotlib.pyplot as plt 

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data() 

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255 

 x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255 

_train = tf.keras.utils.to_categorical(y_train, 10) 

 y_test = tf.keras.utils.to_categorical(y_test, 10) 

model = models.Sequential([ layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), layers.MaxPooling2D((2, 2)), 

layers.Conv2D(64, (3, 3), activation='relu'), 
layers.MaxPooling2D((2, 2)), 
 
layers.Flatten(), 
layers.Dense(64, activation='relu'), 
layers.Dense(10, activation='softmax') 
  

]) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64) 

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) print(f"Test accuracy: {test_acc:.4f}") 