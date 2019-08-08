# UCR time series dataset loader

This is a simple loader for gait and handwriting data of parkinsons desease patients.

## Install
```
pip install git+https://github.com/walwe/ucrloader.git#egg=ucrloader
```


## Usage

```
loader = Parkloader(park_data_dir)
for name in loader.names:
    data = loader.load(name)
    # Access data
    print(data.train_labels)
    print(data.train_data)
    print(data.test_labels)
    print(data.test_data)
```
