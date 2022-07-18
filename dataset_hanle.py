from pyts import datasets
import numpy as np
from sklearn.utils import Bunch

# there are two ways to handle the dataset,the first way is to fetch it from the internet
# the second way is to
def fetch_all_dataset():
    dataset_list = datasets.ucr_dataset_list()
    for dataset in dataset_list:
        datasets.fetch_ucr_dataset(dataset, return_X_y=True)

def _load_ucr_dataset(dataset, path):
    """Load a UCR data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Padded values are represented as NaN's.

    """
    new_path = path + dataset + '/'
    try:
        with(open(new_path + dataset + '.txt', encoding='utf-8')) as f:
            description = f.read()
    except UnicodeDecodeError:
        with(open(new_path + dataset + '.txt', encoding='ISO-8859-1')) as f:
            description = f.read()
    try:
        data_train = np.genfromtxt(new_path + dataset + '_TRAIN.txt')
        data_test = np.genfromtxt(new_path + dataset + '_TEST.txt')

        X_train, y_train = data_train[:, 1:], data_train[:, 0]
        X_test, y_test = data_test[:, 1:], data_test[:, 0]

    except IndexError:
        train = loadarff(new_path + dataset + '_TRAIN.arff')
        test = loadarff(new_path + dataset + '_TEST.arff')

        data_train = np.asarray([train[0][name] for name in train[1].names()])
        X_train = data_train[:-1].T.astype('float64')
        y_train = data_train[-1]

        data_test = np.asarray([test[0][name] for name in test[1].names()])
        X_test = data_test[:-1].T.astype('float64')
        y_test = data_test[-1]

    try:
        y_train = y_train.astype('float64').astype('int64')
        y_test = y_test.astype('float64').astype('int64')
    except ValueError:
        y_train = y_train.astype(str)
        y_test = y_test.astype(str)

    bunch = Bunch(
        data_train=X_train, target_train=y_train,
        data_test=X_test, target_test=y_test,
        DESCR=description,
        url=("http://www.timeseriesclassification.com/"
             "description.php?Dataset={}".format(dataset))
    )

    return bunch

