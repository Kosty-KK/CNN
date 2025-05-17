import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import os
import cv2


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(y_train) # проверяем данные


x_dataset = tf.data.Dataset.from_tensor_slices(x_train)
y_dataset = tf.data.Dataset.from_tensor_slices(y_train)

def process_x(x):
    img = tf.cast(x, tf.float32) / 255
    return img

def process_y(y):
    y = tf.one_hot(y, 10)
    return y

x_dataset = x_dataset.map(process_x)
y_dataset = y_dataset.map(process_y)
dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
dataset = dataset.shuffle(1000)
dataset = dataset.batch(64)
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)


def imshow():
    n = 10
    plt.figure(figsize=(10, 6))
    for images, labels in dataset.take(1):
        for i in range(n):
            img = images[i]
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(img)
            plt.axis('off')
            # cmap='gist_gray'
            ax.get_yaxis().set_visible(False)
        plt.show()
imshow()




inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
x = Dense(128, activation = 'relu')(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(10, activation = 'sigmoid')(x)
outputs = x
simple_nn = keras.Model(inputs, outputs)


class Model(tf.keras.Model):
    def __init__(self, nn):
        super(Model, self).__init__()
        self.nn = nn

    def get_loss(self, y, preds):
        loss = tf.keras.losses.CategoricalCrossentropy()(y, preds)
        return loss
    @tf.function
    def training_step(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.nn(x)
            loss = self.get_loss(y, preds)

        gradients = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))
        return tf.reduce_mean(loss)
    

model = Model(simple_nn)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))


for x, y in dataset.take(1):
    print(model.training_step(x, y))




hist = np.array(np.empty([0]))
epochs = 100
for epoch in range (1, epochs +1):
    loss = 0
    for step, (x, y) in enumerate(dataset):
        loss += model.training_step(x, y)
    clear_output(wait=True)
    print(epoch)
    hist = np.append(hist, loss)
    plt.plot(np.arange(0, len(hist)), hist)
    plt.show()


def imshow_and_pred():
    n = 10
    plt.figure(figsize=(10, 6))
    for images, labels in dataset.take(1):
        for i in range(n):
            img = images[i]
            img_tensor = tf.expand_dims(img, axis = 0)
            pred = model.nn(img_tensor)
            pred = tf.squeeze(pred, axis = 0)
            pred = pred.numpy()

            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(img)
            plt.axis('off')
            ma = pred.max()
            res = np.where(pred == ma)

            plt.title(res[0][0])
            plt.axis('off')
            ax.get_yaxis().set_visible(False)
    plt.show()

# проверяем предсказания модели
imshow_and_pred()  

# сохраняем модель
model.nn.save("my_model.h5")


canvas = np.zeros((280, 280, 3), dtype=np.uint8)

cv2.namedWindow('To finish, press "q"')

def draw_circle(event, x, y, flags, param):
    global drawing_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_mode = True
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing_mode:        
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_mode = False

# Устанавливаем обработчик событий мыши
cv2.setMouseCallback('To finish, press "q"', draw_circle)

# Флаг для отслеживания состояния кнопки мыши
drawing_mode = False

while True:
    # Показываем изображение на экране
    cv2.imshow('To finish, press "q"', canvas)
    
    # Ждем нажатия клавиши
    key = cv2.waitKey(1) & 0xFF
    # Если нажата клавиша 's', сохраним изображение в переменную
    if key == ord('s'):
        my_img = canvas.copy()  # Копируем изображение в новую переменную
        print("Изображение сохранено в переменную 'my_img'.")
        my_img = my_img
        img_tensor = tf.expand_dims(my_img, axis = 0)
        img_tensor = tf.cast(img_tensor, tf.float32) /255
        img_tensor = tf.image.resize(img_tensor, (28, 28), method = 'area')
        img_numpy = img_tensor.numpy()
        img_numpy = img_numpy[:,:,:,0]

        pred = model.nn(img_numpy)
        pred = tf.squeeze(pred, axis = 0)
        pred = pred.numpy()

        ma = pred.max()
        res = np.where(pred == ma)
        print(res[0][0])
    
    # Если нажата клавиша 'c', очищаем холст
    elif key == ord('c'):
        canvas[:, :, :] = 0
    
    # Если нажата клавиша 'q', выходим из цикла
    elif key == ord('q'):
        break

# Закрываем все окна
cv2.destroyAllWindows()
