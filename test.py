# coding: utf-8
# @Author: Salt
import matplotlib.pyplot as plt
a = [1,2,3]
b = [4,5,6]
plt.plot(a,b)
plt.show()
# import numpy as np
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import nibabel as nib
# from nibabel import nifti1
# from nibabel.viewers import OrthoSlicer3D
#
# example_filename = 'imaging.nii.gz'
# img = nib.load(example_filename)
# width, height, queue = img.header.get_data_shape()
# print(width, height, queue)
#
# # array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
# # affine = np.diag([1, 2, 3, 1])
# # array_img = nib.Nifti1Image(array_data, affine)
# print(img.dataobj)
# print(type(img.dataobj))
# # print(img.dataobj[300,:,:][300])
# image_data = img.get_fdata()
# print(type(image_data))
# print(image_data[300][300])