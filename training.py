import numpy as np
from architecture import EIT
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

batchSize = 250
epochs = 1000
optimizer = "adam"
lossFunction = "mse"
patience = 20
earlyStop = EarlyStopping(monitor = "val_loss" ,patience = patience)

print("Importing dataset...")
inputData = np.load("../data/711_input.npy")
inputData = np.reshape(inputData, (len(inputData), 32, 32, 1))
outputData = np.load("../data/711_output.npy")
print(f"input Data: {inputData.shape}")
print(f"output Data: {outputData.shape}")

xTrain, xTest, yTrain, yTest = train_test_split(inputData, outputData,
												test_size = 499)
print(f"xTrain: {xTrain.shape}")
print(f"xTest: {xTest.shape}")
print(f"yTrain: {yTrain.shape}")
print(f"yTest: {yTest.shape}")

model = EIT().build()
model.compile(optimizer = optimizer,
			  loss = lossFunction,
			  metrics = ["mse"])
model.summary()

history = model.fit(x = xTrain, y = yTrain,
					validation_data = (xTest, yTest),
					epochs = epochs,
					batch_size = batchSize,
					verbose = 1,
					callbacks = [earlyStop])

print("Importing validation data...")
xValidation = np.load("../data/1366_seed_A_input.npy")
xValidation = np.reshape(xValidation, (len(xValidation), 32, 32, 1))
yValidation = np.load("../data/1366_seed_A_output.npy")
print(f"input Validation Data: {xValidation.shape}")
print(f"output Validation Data: {yValidation.shape}")

score = model.evaluate(x = xValidation, y = yValidation)
print(f"{model.metrics_names[0]}: {score[0]}")
print()

print("Saving model...", end = " ")
model.save("eit.h5")
print("Done")

del model