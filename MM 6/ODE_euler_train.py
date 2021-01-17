import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import RNN, Dense, Layer
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.framework import tensor_shape
from tensorflow import float32, concat, convert_to_tensor
from tensorflow.keras.callbacks import ModelCheckpoint


class EulerCell(Layer):
    def __init__(self, fLayer, x0=None, units=1, **kwargs):
        super(EulerCell, self).__init__(**kwargs)
        self.units = units
        self.x0 = x0
        self.fLayer = fLayer
        self.state_size = tensor_shape.TensorShape(self.units)
        self.output_size = tensor_shape.TensorShape(self.units)

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs, states):
        inputs = convert_to_tensor(inputs)
        x_tm1 = convert_to_tensor(states)
        mm_d_tm1 = concat((inputs, x_tm1[0, :]), axis=1)
        df_t = self.fLayer(mm_d_tm1)
        x = df_t + x_tm1[0, :]
        return x, [x]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return self.x0


class Normalization(Layer):
    def __init__(self, t_low, t_up, x_low, x_up, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.low_bound_t = t_low
        self.upper_bound_t = t_up
        self.low_bound_x = x_low
        self.upper_bound_x = x_up

    def build(self, input_shape, **kwargs):
        self.built = True

    def call(self, inputs):
        output = (inputs - [self.low_bound_t, self.low_bound_x]) / [(self.upper_bound_t - self.low_bound_t),
                                                                    (self.upper_bound_x - self.low_bound_x)]
        return output


def create_model(x0, fLayer, batch_input_shape, return_sequences=False, return_state=False):
    euler = EulerCell(fLayer=fLayer, x0=x0, batch_input_shape=batch_input_shape)
    PINN = RNN(cell=euler, batch_input_shape=batch_input_shape, return_sequences=return_sequences,
               return_state=return_state)
    model = Sequential()
    model.add(PINN)
    model.compile(loss='mse', optimizer=RMSprop(1e-2))
    return model


if __name__ == "__main__":
    t_train = np.asarray(pd.read_csv('./data/ttrain.csv'))[:, :, np.newaxis]
    xt_train = np.asarray(pd.read_csv('./data/xttrain.csv'))
    x0 = np.asarray(pd.read_csv('./data/x0.csv'))[0, 0] * np.ones((t_train.shape[0], 1))

    fLayer = Sequential()
    fLayer.add(Normalization(np.min(t_train), np.max(t_train), np.min(xt_train), np.max(xt_train)))
    fLayer.add(Dense(5, activation='tanh'))
    fLayer.add(Dense(1))

    t_range = np.linspace(np.min(t_train), np.max(t_train), 1000)
    xt_range = np.linspace(np.min(xt_train), np.max(xt_train), 1000)[np.random.permutation(np.arange(1000))]
    f_range = - t_range**3 * 4 * 2.7**(-t_range**3 / 3)

    fLayer.compile(loss='mse', optimizer=RMSprop(1e-2))
    inputs_train = np.transpose(np.asarray([t_range, xt_range]))
    fLayer.fit(inputs_train, f_range, epochs=200)

    model = create_model(x0=convert_to_tensor(x0, dtype=float32), fLayer=fLayer,
                         batch_input_shape=t_train.shape)
    model.fit(t_train, xt_train, epochs=200, steps_per_epoch=1, verbose=1)