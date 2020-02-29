from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten

class lenet:
	@staticmethod
	def build(inputShape = (32,32,1), numberOfCondValues = 710):
		model = Sequential()

		model.add(Conv2D(filters = 6, 
						 kernel_size = (5,5),
					  	 strides = (1,1),
					  	 padding = "same",
					  	 activation = "relu",
					  	 input_shape = inputShape))

		model.add(MaxPool2D(pool_size = (2,2),
							strides = (2,2)))

		model.add(Conv2D(filters = 16,
						 kernel_size = (5,5),
						 strides = (1,1),
						 padding = "same",
						 activation = "relu"))

		model.add(MaxPool2D(pool_size = (2,2),
							strides = (2,2)))

		model.add(Flatten())

		model.add(Dense(units = 120,
						activation = "relu"))

		model.add(Dense(units = 84,
						activation = "relu"))

		model.add(Dense(units = numberOfCondValues,
						activation = "relu"))

		return model