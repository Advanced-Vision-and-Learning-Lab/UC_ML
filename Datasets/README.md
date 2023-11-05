# Downloading datasets:

Note: Due to the size of the datasets, the generator information were not 
uploaded to the repository. For the paper, the 7000-bus case was used, but smaller test cases (14-bus and 600-bus) are also available to evaluate each classifier. 
Please follow the following instructions
to ensure the code works. If the dataset is used,
please cite the appropiate source (paper, repository, etc.) as mentioned
on the webpage and provided here.

##  Unit Commitment Machine Learning Datasets

Please download the [`UC_ML dataset`](https://drive.google.com/drive/folders/11uTp2SuZGi_0SnKx2fxPyQPkJgO-mSO2?usp=sharing)
and follow these instructions:

1. Download the desired bus cases in the `Datasets` folder
2. The structure of the `Dataset` folder is as follows:
```
Datasets/
    ├── 14 Bus/
    │   ├── Features.csv
    │   ├── Labels.csv
    ├── 600 Bus/
    │   ├── Features.csv
    │   ├── Labels.csv
    ├── 7000 Bus/
    │   ├── 7000 Bus Generator Labels.csv
    │   ├── Features.csv
    │   ├── Labels.csv
```
3. After downloading the desired bus case(s), please run [`demo.py`](https://github.com/Advanced-Vision-and-Learning-Lab/UC_ML/blob/main/demo.py) to run the multi-classification experiments.

If you use the UC_ML dataset, please cite the following reference using the following entry.

**Plain Text:**

F. Safdarian, J. Peeples, D. Richards, J. Wert, and T. Overbye, "A Fast Learning-Based Unit Commitment Strategy with AC Optimal Power Flow for Large Grids with Direct Inclusion of Weather,"  in Review.

**BibTex:**

```
@inproceedings{Safdarian2023fast,
  title={A Fast Learning-Based Unit Commitment Strategy with AC Optimal Power Flow for Large Grids with Direct Inclusion of Weather},
  author={Safdarian, Farnaz and Peeples, Joshua, and Richards, Dillon, and  Wart, Jessica, and Overbye, Thomas},
  booktitle={TBD},
  pages={TBD},
  year={2023},
  organization={TBD}
}
```
