from pathlib import Path
import logging
import pandas as pd
import numpy as np

HC = 'HC'
PA = 'Patients'
GAIT = 'gaitRaw'
HW = 'handwriting'


class ParkLoader(object):
    """Loads a parkinson data folder"""
    def __init__(self, path):
        """
        :param path: Path to mulltimodal parkinsons dataset
        possible splits: handwriting_all, gait_all, handwriting_by_word, gait_by_distance
        Text files in it are in the format PatientID_TaskID.txt
        """
        self._path = Path(path)
        if not self._path.is_dir():
            logging.error("{} is not a directory.".format(path))
            raise NotADirectoryError("Given parkinsons data directory does not exist")

        self._files = list(self._path.glob("**/*.txt"))

        for n in self.names:
            self._data_sets = {n: f for f in self._files if n in str(f.stem)}

    @property
    def names(self):
        """
        Dataset names
        """
        
        return list(set(f.stem.split('_')[-1] for f in self._files))

    def load(self, name, z_norm=False):
        """
        Load dataset from disk
        :param name: name of dataset
        :return: UCRData object
        """
        try:
            return ParkData.from_paths(self._data_sets.get(name), z_norm)
        except KeyError:
            logging.error("Data{} not available in {}".format(name, self._path))


class ParkData(object):
    """Can be either gait or handwriting data"""
    def __init__(self, name: str, train_data, train_labels, test_data, test_labels):
        """
        :param name: Name of task
        :param train_data: train data array
        :param test_data: test data array
        """
        self.name = name
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels

    @classmethod
    def from_paths(cls, paths: list, z_norm=False):
        """
        Load single dataset add given path
        :param z_norm: Z-normalize data?
        :param paths: list containing paths matching a single dataset
        :return: ParkData
        """

        name = paths[0].name.split('_')[-1]
        test_fn = path / "{}_TEST".format(name)
        train_fn = path / "{}_TRAIN".format(name)

        logging.info("Loading {}".format(name))
        train_data = cls._read_file(train_fn)
        test_data = cls._read_file(test_fn)

        train_labels, train_data = cls._split(train_data)
        test_labels, test_data = cls._split(test_data)

        if z_norm:
            train_data, test_data = cls._z_norm(train_data, test_data)

        return cls(name, train_data, train_labels, test_data, test_labels)

    @staticmethod
    def _split(data):
        """Split into labels and values. First column are labels"""
        labels = np.array(list(
            map(int, data.values[:, 0])
        ))
        values = data.values[:, 1:]
        return labels, values

    @staticmethod
    def _read_file(file_name):
        df = pd.read_csv(file_name, header=None)
        return df

    @staticmethod
    def _z_norm(train_data, test_data):
        axis = 0
        ddof = 0

        # Calculate mean and std on train data
        mns = np.nanmean(train_data)
        sstd = np.nanstd(train_data)

        def norm(a):
            a = np.asanyarray(a)
            if axis and mns.ndim < a.ndim:
                return ((a - np.expand_dims(mns, axis=axis)) /
                        np.expand_dims(sstd, axis=axis))
            else:
                return (a - mns) / sstd

        return norm(train_data), norm(test_data)

