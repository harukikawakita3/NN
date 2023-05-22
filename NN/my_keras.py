# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# model = Sequential()
# model.add(Dense(10, input_dim=5, activation='relu'))
# model.add(Dense(5, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# x_train = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
# y_train = np.array([[1, 0, 0, 0, 0], [0, 0, 1, 0, 0]])

# model.fit(x_train, y_train, epochs=100, batch_size=32)

# x_test = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
# y_pred = model.predict(x_test)

# print(y_pred)


# # keras単純CNN

# # 必要なライブラリをインポート
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# # Sequentialモデルを使用して新しいモデルを初期化
# model = Sequential()

# # 最初の畳み込み層を追加: 32の出力フィルタ、3x3のカーネルサイズ、ReLU活性化関数、入力形状は28x28ピクセルの1チャンネル画像
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# # 最初のプーリング層を追加: 2x2のプーリングサイズ
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # 2つ目の畳み込み層を追加: 64の出力フィルタ、3x3のカーネルサイズ、ReLU活性化関数
# model.add(Conv2D(64, (3, 3), activation='relu'))

# # 2つ目のプーリング層を追加: 2x2のプーリングサイズ
# model.add(MaxPooling2D(pool_size=(2, 2)))

# # データを1次元配列に変換
# model.add(Flatten())

# # 全結合層を追加: 128の出力ユニット、ReLU活性化関数
# model.add(Dense(128, activation='relu'))

# # 出力層を追加: 10の出力ユニット（MNISTの数字0-9を表現）、ソフトマックス活性化関数
# model.add(Dense(10, activation='softmax'))

# # モデルの概要を表示
# model.summary()