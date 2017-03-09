from rnn import RNN
import numpy as np
import itertools

nBatch = 500
batchSize = 20


rnn = RNN(inputDim=2, stateDim=19)

for i in range(nBatch):
    x = [0,0,1]
    x = list(itertools.islice(itertools.cycle(x), batchSize - batchSize%len(x)+1))
    x = np.array(x)

    rnn.update(x)

print(rnn.predict(start=0, nStep=10))
