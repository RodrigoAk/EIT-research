from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten

class EIT:
	@staticmethod
	def build(inputShape = (32,32,1), numberOfCondValues = 710):
		model = keras.models.Sequential()

		model.add(Conv2D(filters = 30, 
						 kernel_size = (5,5),
					  	 strides = (1,1),
					  	 padding = "same",
					  	 activation = "relu",
					  	 input_shape = inputShape))
		model.add(MaxPool2D(pool_size = (2,2),
							strides = (2,2)))

		model.add(Conv2D(filters = 60,
						 kernel_size = (5,5),
						 strides = (1,1),
						 padding = "same",
						 activation = "relu"))
		model.add(MaxPool2D(pool_size = (2,2),
							strides = (2,2)))

		model.add(Flatten())

		model.add(Dense(units = numberOfCondValues//2,
						activation = "relu"))

		model.add(Dense(units = numberOfCondValues,
						activation = "relu"))

		return model