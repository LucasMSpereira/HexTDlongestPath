import tensorflow as tf
import utilities

ds = utilities.dataManager(0, 10, 10)
d = ds.readHDF5file("dataset.hdf5")