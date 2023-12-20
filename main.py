import numpy as np
import pandas as pd
import tkinter as tk
from keras.layers import Input, Dense, Flatten
from keras.models import Sequential, Model
from keras.losses import sparse_categorical_crossentropy
from keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time


train = pd.read_csv("mnist_train.csv").values
Y_train = train[:, 0]
X_train = train[:, 1:]

test = pd.read_csv("mnist_test.csv").values
Y_test = test[:, 0]
X_test = test[:, 1:]

X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)


x = Input(shape=(28, 28))
flatten_x = Flatten()(x)
h1 = Dense(64, activation="relu")(flatten_x)
h2 = Dense(64, activation="relu")(h1)
h3 = Dense(64, activation="relu")(h2)
out = Dense(10, activation="softmax")(h3)
model = Model(inputs=x, outputs=out)

# Compile the model
opt = Adam(learning_rate=0.001)
model.compile(
    optimizer=opt,
    loss=sparse_categorical_crossentropy,
    metrics=[sparse_categorical_accuracy],
)

# Train the model
bs = 64
n_epoch = 10
model.fit(
    X_train,
    Y_train,
    batch_size=bs,
    epochs=n_epoch,
    validation_data=(X_test, Y_test),
)

# Create a window
root = tk.Tk()
root.title("Digit Prediction")

# Create Canvas to figure
canvas = FigureCanvasTkAgg(plt.Figure(), master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Get the axes
ax = canvas.figure.add_subplot(111)


def update_digit(image, label):
    ax.clear()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Predicted: {label}")
    canvas.draw()

# Set interactive mode
plt.ion()
plt.show()

# Predict and update the figure for each test example
for i in range(len(X_test)):
    image = X_test[i]
    real_label = Y_test[i]

    # Predict the digit
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    predicted_label = np.argmax(prediction)

    # Update the figure
    update_digit(image, predicted_label)

    # Pause for a short duration
    root.update_idletasks()
    root.update()

    # Pause for 1.5 seconds
    time.sleep(1.5)


plt.ioff()
plt.show()
root.mainloop()
