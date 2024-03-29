# Result
### Result from 3D_sclier
![image](result/seg_predict.png)
### Result GUI from scratch
![image](result/res_gui.png)

# Data --> KiTS19

Here is the official [2019 KiTS Challenge](https://kits19.grand-challenge.org) repository.

## Usage

To get the data, please clone this repository, and then run `get_imaging.py`. For example
```text
git clone https://github.com/chi-sq/3DUNet
cd 3DUNet
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```
This will download the much larger and static image files from a separate source. The `data/` directory should then be structured as follows

```
data
├── case_00000
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
├── case_00001
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
...
├── case_00209
|   ├── imaging.nii.gz
|   └── segmentation.nii.gz
└── kits.json
```
And the you can divide them to train and validation datesets.

###  Data description
In the segmentation, a value of 0 represents background, 1 represents kidney, and 2 represents tumor.

### Tutorial
You can move to ``` 3DUNet_Tutorial.ipynb``` to run this repository.

