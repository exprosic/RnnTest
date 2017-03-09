import numpy as np
from collections import Iterable

def softmax(x):
    e_x = np.exp(x - x.mean())
    return e_x / e_x.sum()

def initWeight(shape):
    #http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    r = np.sqrt(6 / sum(shape))
    return np.random.uniform(-r, r, shape)

class RNN:
    MAX_LENGTH = 40

    def __init__(self, inputDim, stateDim):
        self.inputDim = inputDim
        self.stateDim = stateDim
        self.V = initWeight((inputDim, stateDim))
        self.U = initWeight((stateDim, inputDim))
        self.W = initWeight((stateDim, stateDim))
        self.dE_dV = np.empty(self.V.shape)
        self.dE_dU = np.empty(self.U.shape)
        self.dE_dW = np.empty(self.W.shape)
        #self.s0 = np.random.uniform(-1, 1, size=(stateDim,))
        self.s0 = np.zeros((stateDim,))

    def update(self, x):
        assert 0 < len(x) < self.MAX_LENGTH
        assert all(0<=xi<self.inputDim and not isinstance(xi, Iterable) for xi in x)
        self.x = x
        self.s = {0: self.s0}
        self.y = {}

        self._propagateForward()
        self._propagateBackward()

        np.clip(self.dE_dV, -1, 1, out=self.dE_dV)
        np.clip(self.dE_dU, -1, 1, out=self.dE_dU)
        np.clip(self.dE_dW, -1, 1, out=self.dE_dW)

        self.V -= self.dE_dV
        self.U -= self.dE_dU
        self.W -= self.dE_dW

    def _propagateForward(self):
        #x: data sequence
        #input: x[0..-2]
        #label: x[1..-1]
        #s[i] = tanh(so = U*x[i-1] + W*s[i-1])
        #y[i] = softmax(yo = V*s[i])

        for i in range(1, len(self.x)):
            #x[0..-2], s[1..-1], y[1..-1]
            so = self.U[:,self.x[i-1]] + self.W.dot(self.s[i-1]) #U[:,x] = U.dot(onehot(x))
            self.s[i] = np.tanh(so)
            yo = self.V.dot(self.s[i])
            self.y[i] = softmax(yo)

    def _propagateBackward(self):
        self.dE_dV.fill(0)
        self.dE_dU.fill(0)
        self.dE_dW.fill(0)
        dE_ds_i = np.zeros(self.s[0].shape)

        for i in reversed(range(1, len(self.x))):
            #y[-1..1], s[-1..1]
            target = self.x[i]

            dE_dyo_i = self.y[i]
            dE_dyo_i[target] -= 1

            dEi_ds_i = dE_dyo_i.dot(self.V) #dyo_ds[i] = V
            dE_ds_i += dEi_ds_i
            dE_dso_i = dE_ds_i * (1 - self.s[i]**2) # = dEi_ds_i.dot(tanh'(so[i]))

            self.dE_dV += np.outer(dE_dyo_i, self.s[i])
            self.dE_dU[:,self.x[i]] += dE_dso_i #dE_dU += np.outer(dE_ds_i, onehot(x[i]))
            self.dE_dW += np.outer(dE_dso_i, self.s[i-1])

            dE_ds_i = dE_ds_i.dot(self.W) #dE[i:]_ds[i-1] = dE[i:]_ds[i] * ds[i]_ds[i-1] (=W)

    def predict(self, start, nStep):
        s = self.s0
        result = [start]

        for i in range(nStep):
            s = np.tanh(self.U[:,result[-1]] + self.W.dot(s))
            y = softmax(self.V.dot(s))
            result.append(np.argmax(y))

        return result

