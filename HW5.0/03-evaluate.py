from keras.models import load_model
import numpy as np
model = load_model('model')
value = 'cnn'
import pickle

f = open('xt', 'rb')
X_test = pickle.load(f)
f.close()
f = open('yt', 'rb')
ytest = pickle.load(f)
f.close()

# evaluate
if value == 'cnn':
    loss, acc = model.evaluate(X_test, np.array(ytest))

if value == 'rnn':
    loss, acc = model.evaluate(X_test.reshape([X_test.shape[0], 50, 1]).astype(np.float32), np.array(ytest))
print('Test Accuracy: %f' % (acc*100))