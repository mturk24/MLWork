from mnist import MNIST
import sklearn.metrics as metrics
import numpy as np
import scipy
import math
import csv

NUM_CLASSES = 10

def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return (X_train, labels_train), (X_test, labels_test)


def train(X_train, y_train, reg=0):
    ''' Build a model from X_train -> y_train '''
    print X_train.shape, "shape of x model"
    print y_train.shape, "shape of y model"
    dot_X = X_train.T.dot(X_train)     
    return scipy.linalg.solve(dot_X + reg * np.eye(dot_X.shape[0]), X_train.T.dot(y_train), sym_pos=True)


def train_gd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using batch gradient descent '''
    # print X_train, "this is the X model"
    # print y_train, "this is the Y model"
    weight = np.ones((X_train.shape[1], NUM_CLASSES))    
    dot_X = X_train.T.dot(X_train)  
    dot_Y = X_train.T.dot(y_train)  
    for i in range(num_iter):
        change = alpha*(1.0/60000)*(dot_X.dot(weight) - dot_Y + reg*weight)
        weight = weight - change
    return weight

def train_sgd(X_train, y_train, alpha=0.1, reg=0, num_iter=10000):
    ''' Build a model from X_train -> y_train using stochastic gradient descent '''
    weight = np.ones((X_train.shape[1], NUM_CLASSES))    
    for i in range(num_iter):
        j = np.random.randint(0, X_train.shape[0])
        x = np.resize(np.array(X_train[j,:]), (1500, 1))
        y = np.resize(np.array(y_train[j,:]), (10, 1))
        dot_X = x.dot(x.T)
        dot_W = dot_X.dot(weight)
        dot_Y = x.dot(y.T) 
        change = alpha*(dot_W - dot_Y + reg*weight)
        weight = weight - change
    return weight
def one_hot(labels_train):
    '''Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10} '''
    # oneHotMatrix = np.zeros((labels_train.shape[0], NUM_CLASSES))
    # for point in range(labels_train.shape[0]):
    #     oneHotMatrix[point][labels_train[point]] = 1
    # return oneHotMatrix
    return np.eye(NUM_CLASSES)[labels_train]

def predict(model, X):
    ''' From model and data points, output prediction vectors '''
    return np.argmax(X.dot(model), axis=1)

G = np.random.normal(0, 0.2, (1500, 784))
b = np.random.uniform(0, 2 * math.pi, 1500)
def phi(X):
    ''' Featurize the inputs using random Fourier features '''
    dot_X = G.dot(X.transpose()) 
    tiled = np.tile(b, (X.shape[0], 1))
    add_tiled = dot_X + tiled.transpose()
    cosines = np.vectorize(math.cos)
    vectorized_matrix = cosines(add_tiled)
    return vectorized_matrix.transpose()


if __name__ == "__main__":
    (X_train, labels_train), (X_test, labels_test) = load_dataset()
    y_train = one_hot(labels_train)
    y_test = one_hot(labels_test)
    X_train, X_test = phi(X_train), phi(X_test)

    model = train(X_train, y_train, reg=0.1)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Closed form solution")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
    

    model = train_gd(X_train, y_train, alpha=1e-3, reg=0.1, num_iter=20000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Batch gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))

    model = train_sgd(X_train, y_train, alpha=1e-4, reg=0.075, num_iter=100000)
    pred_labels_train = predict(model, X_train)
    pred_labels_test = predict(model, X_test)
    print("Stochastic gradient descent")
    print("Train accuracy: {0}".format(metrics.accuracy_score(labels_train, pred_labels_train)))
    print("Test accuracy: {0}".format(metrics.accuracy_score(labels_test, pred_labels_test)))
    with open('myOutput.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Id',  'Category'])
        for i in range(pred_labels_test.shape[0]):
            spamwriter.writerow([str(i),  str(pred_labels_test[i])])
    
