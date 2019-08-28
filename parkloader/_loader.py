from pathlib import Path
import logging
import pandas as pd
import numpy as np

HC = 'HC'
PA = 'Patients'
GAIT = 'gaitRaw'
HW = 'handwriting'
METADATA = 'metadata.csv'
LABEL = 'label'
ID = 'ID'


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

        def classes(x):
            if x < 21:
                return 1
            elif 20 < x < 40:
                return 2
            else:
                return 3
        hc_ids = sorted(list(set([f.stem.split('_')[0] for f in self._files if HC in f.parts])))
        tmp = pd.read_csv(str(self._path / METADATA))
        tmp[LABEL] = tmp['updrs_total'].apply(classes)
        self._metadata = tmp[[ID, LABEL]]
        for id in hc_ids:
            self._metadata = self._metadata.append({ID: id, LABEL: 0}, ignore_index=True)

        self._data_sets = {}
        for n in self.names:
            self._data_sets[n] = [f for f in self._files if n in str(f.stem)]

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
            return ParkData.from_paths(self._data_sets.get(name), name, self._metadata)
        except KeyError:
            logging.error("Data{} not available in {}".format(name, self._path))


class ParkData(object):
    """Can be either gait or handwriting data"""
    def __init__(self, name: str, data):
        """
        :param name: Name of task
        :param data: dataframe containing data, labels and patient_id needed for splitting
        """
        self.name = name
        self.data = data

    @classmethod
    def from_paths(cls, paths: list, name: str, metadata):
        """
        Load single dataset add given path
        :param paths: list containing paths matching a single dataset
        :param name: name of task to load
        :return: ParkData
        """
        logging.info("Loading {}".format(name))
        header = None if HW in str(paths[0]) else 0
        delim = ';' if HW in str(paths[0]) else ' '
        data = cls._read_files(paths, header, delim, metadata)
        return cls(name, data)

    def leave_one_out(self, z_norm=False):
        for pid in [p for p in self.data[ID].unique() if HC not in p]:
            train = self.data.where(self.data[ID] != pid).dropna()
            test = self.data.where(self.data[ID] == pid).dropna()
            if len(test) < 1:
                continue
            train_data = np.asarray(train['data'].values)
            test_data = np.asarray(test['data'].values)
            train_labels = np.asarray(train[LABEL].values)
            test_labels = np.asarray(test[LABEL].values)
            if z_norm:
                train_data, test_data = self._z_norm(train_data, test_data)
                yield train_data, test_data, train_labels, test_labels, pid
            else:
                yield train_data, test_data, train_labels, test_labels, pid

    @staticmethod
    def _split(data):
        """Split into labels and values. last column are labels"""
        labels = [int(d[1]) for d in data]
        values = [d[0] for d in data]
        return labels, values

    @staticmethod
    def _read_files(paths, header, delim, metadata):
        def get_labels(p):
            pid = p.stem.split('_')[0]
            pid = pid if pid[-1].isdigit() else pid[:-1]
            label = metadata.where(metadata[ID].str.startswith(pid)).dropna()[LABEL].iloc[0]
            return label, pid
        data = []
        for p in paths:
            try:
                df = pd.read_csv(str(p), header=header, sep=delim)
                lbl, pid = get_labels(p)
                # dirty fucking hack: if header is none, then HW data. only first 6 values needed/valid
                vals = df.values[:, 0:6] if header is None else df.values
                data.append((vals, lbl, pid))
            except Exception as e:
                logging.info("Loading {} has failed".format(p))
                logging.error(e)
                continue

        return pd.DataFrame(data, columns=['data', LABEL, ID])

    @staticmethod
    def _z_norm(train_data, test_data):
        axis = 0
        ddof = 0

        # Calculate mean and std on train data
        mns = np.nanmean([v for ts in train_data for v in ts])
        sstd = np.nanstd([v for ts in train_data for v in ts])

        def norm(a):
            a = np.asanyarray(a)
            if axis and mns.ndim < a.ndim:
                return ((a - np.expand_dims(mns, axis=axis)) /
                        np.expand_dims(sstd, axis=axis))
            else:
                return (a - mns) / sstd

        return np.asarray([norm(t) for t in train_data]), np.asarray([norm(t) for t in test_data])

