# Parkinson time series dataset loader

This is a simple loader for gait and handwriting data of parkinson disease patients.

## Install
```
pip install git+https://github.com/patientzero/parkloader.git#egg=parkloader
```


## Usage

```
loader = Parkloader(park_data_dir)
for name in loader.names:
    data = loader.load(name)
    # Access data (pandas Dataframe)
     for train, test, train_lbls, test_lbls, pid in data.leave_one_out(True):
            assert(len(train) == len(train_lbls))
            assert(len(test) == len(test_lbls))
            print(pid)
    
```
