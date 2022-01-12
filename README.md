# Exploratory_data_analysis 

Data preprocessing of combination of numerical and categorical data for deep learning network training.
The raw input file in either .csv or .xlsx is preprocessed with categorical data ordinal encoding and the processed
data is saved in .csv for use in deep learning training.

* The script automatically compute numerical columns and categorical columns, as well as the mapping of the categorical ordinal
encoded values and the correspoding categorical values are stored as **.npy and .npz** files respectively.

* The numerical columns values can be loaded from xxxx_numerical_col.npy using:
numerical_cols = numpy.load(xxx_numerical_col.npy)
The cateogorical columns with mapping of each column values to corresponding ordinal values can be loaded with:
categorical_cols = numpy.load(xxx_categorical_col.npz, allow_pickle=True)
categorical_cols = categorical_cols.files

* The categorical columns values mapping to each ordinal encoding can be retirved as a dictionary:
categorical_cols = numpy.load(xxx_categorical_col.npz, allow_pickle=True)
for cat_col in categorical_cols:
cat_dict = categorical_cols[cat_col]

print(cat_dict)

- [ ] Add sample data files
