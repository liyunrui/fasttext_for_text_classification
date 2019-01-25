import logging
import os
from datetime import datetime
import itertools
from sklearn.metrics import confusion_matrix
# plt
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [16,9]

#--------------------
# logging config
#--------------------

def init_logging(log_dir):
    '''
    for recording the experiments.
    log_dir: path
    '''
    #--------------
    # setting
    #--------------
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_file = 'log_{}.txt'.format(date_str)
    #--------------
    # config
    #--------------
    logging.basicConfig(
        filename = os.path.join(log_dir, log_file),
        level = logging.INFO,
        format = '[[%(asctime)s]] %(message)s',
        datefmt = '%m/%d/%Y %I:%M:%S %p'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

#--------------------
# confusion matrix
#--------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def confusion_matrix_wrapper(y_true, y_pred, labels):
    """
    Interface for plot confusion matrix.

    y_true: list-like. The element is integer index representing a certain class.
    y_pred: list-llke.
    labels: list-like. The element is string represetning name of class.
    Let's say we have 3 classes.

    y_true = [2, 0, 2, 2, 0, 1]. 
    y_pred = [0, 0, 2, 2, 0, 2].
    labels = ["ant", "bird", "cat"].
    2 : cat
    0 : ant
    1 : berd
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

