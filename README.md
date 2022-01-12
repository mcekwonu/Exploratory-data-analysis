# Exploratory_data_analysis

Data preprocessing of combination of numerical and categorical data for deep learning network training.
The raw input file in either .csv or .xlsx is preprocessed with categorical data ordinal encoding and saved in .csv for further use in deep learning embedding network.

* The script automatically compute numerical columns and categorical columns, as well as the mapping of the categorical ordinal
encoded values and the correspoding categorical values are stored as **.npy and .npz** files respectively.

* The numerical columns values can be loaded from xxxx_numerical_col.npy using:
```
numerical_cols = numpy.load(xxx_numerical_col.npy)
```
* The cateogorical columns with mapping of each column values to corresponding ordinal values can be loaded with:
```
categorical_cols = numpy.load(xxx_categorical_col.npz, allow_pickle=True)
categorical_cols = categorical_cols.files
```

* The categorical columns values mapping to each ordinal encoding can be retirved as a dictionary:
```
categorical_cols = numpy.load(xxx_categorical_col.npz, allow_pickle=True)

for cat_col in categorical_cols:
    cat_dict = categorical_cols[cat_col]

print(cat_dict)
```
### HOW TO USE:
* Download the script exploratory_analysis.py and run via terminal or cmd:
```
$ python -m exploratory_analysis --data_path xxxx.csv --save_dir xxxx
```


## Highlights:
1. Automatically encode all categorical data.
2. Save processsed combined numerical and categorical data.
3. Saves dictionary of categorical values and ordinal encodings and categorical columns.
4. Saves list of numerical columns present.



**TO DO**

- [ ] Add python file
- [ ] Add sample data file
