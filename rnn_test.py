from rnn import RNN
import numpy as np
import itertools

#nBatch = 500
#batchSize = 20
#
#
#rnn = RNN(inputDim=2, stateDim=19)
#
#for i in range(nBatch):
#    x = [0,0,1]
#    x = list(itertools.islice(itertools.cycle(x), batchSize - batchSize%len(x)+1))
#    x = np.array(x)
#
#    rnn.update(x)
#
#print(rnn.predict(start=0, nStep=10))

def run01test(nBatch, batchSize, stateDim, pattern, predictLength):
    assert len(pattern) <= batchSize
    x = list(itertools.islice(itertools.cycle(pattern), batchSize - batchSize%len(pattern) + 1))
    x = np.array(x)

    rnn = RNN(inputDim=2, stateDim=stateDim)

    for i in range(nBatch):
        rnn.update(x)

    print(rnn.predict(start=pattern[0], nStep=predictLength))


configurations = {}
configurations['01'] = dict(nBatch=200, batchSize=20, stateDim=5, pattern=[0,1], predictLength=10)
configurations['010'] = dict(nBatch=200, batchSize=20, stateDim=29, pattern=[0,1,0], predictLength=10)
configurations['0011'] = dict(nBatch=500, batchSize=20, stateDim=40, pattern=[0,0,1,1], predictLength=10)
configurations['0111'] = dict(nBatch=500, batchSize=20, stateDim=35, pattern=[0,1,1,1], predictLength=10)
configurations['1101 failed'] = dict(nBatch=10000, batchSize=20, stateDim=40, pattern=[1,1,0,1], predictLength=10)
configurations['1101'] = dict(nBatch=10000, batchSize=20, stateDim=50, pattern=[1,1,0,1], predictLength=10)
configurations['1101.a'] = dict(nBatch=2000, batchSize=20, stateDim=60, pattern=[1,1,0,1], predictLength=10)
configurations['1101.b'] = dict(nBatch=20000, batchSize=20, stateDim=70, pattern=[1,1,0,1], predictLength=10)

run01test(**configurations['1101.a'])
