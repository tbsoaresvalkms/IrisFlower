import pandas
import numpy as np
from keras import utils
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

np.random.seed(7)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
dataset = dataset.values

target = dataset[:, -1]
features = dataset[:, :-1]

encoder = LabelEncoder()
encoder.fit(target)
encoded_Y = encoder.transform(target)

target_flowers = utils.to_categorical(encoded_Y)
features_flowers = features.astype('float32')

scale = MinMaxScaler(feature_range=(0, 1))
features_flowers = scale.fit_transform(features_flowers)


def baseline_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(4,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=3, verbose=1)

X_train, X_test, Y_train, Y_test = train_test_split(features_flowers, target_flowers, test_size=0.33, random_state=7)

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X_train, Y_train, cv=kfold)

estimator.fit(X_train, Y_train)
Y_predicitions = estimator.predict(X_test)
flowers_prediciton = encoder.inverse_transform(Y_predicitions)
flowers_correct = encoder.inverse_transform(Y_test.argmax(axis=1))

correct = sum((flowers_correct == flowers_prediciton))
total = len(flowers_correct)
hit_rate = correct / total

print("\nHit rate: %.2f%%" % (hit_rate * 100))
print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
