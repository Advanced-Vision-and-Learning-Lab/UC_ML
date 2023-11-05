# Downloading models:

Note: Due to the size of the models , the models' weights were not 
uploaded to the repository. For the paper, the 7000-bus case was used, so the saved models for the 7000-bus case are available. 
Please follow the following instructions
to ensure the code works. If the models are used,
please cite the appropiate source (paper, repository, etc.) as mentioned
on the webpage and provided here.

##  Unit Commitment Machine Learning Models

Please download the [`UC_ML models`](https://drive.google.com/drive/folders/1HP19lAyvtMpHGdE7CVv_0yI2A5i4KtQR?usp=sharing)
and follow these instructions:

1. Download the desired pretrained models in the `Pretrained_Models` folder
2. The structure of the `Pretrained_Models` folder is as follows:
```
Pretrained_Models/
    ├── KNN.sav
    ├── MLP.sav
    ├── Random Forest.sav
```
3. After downloading the desired model(s), please run [`demo.py`](./main/demo.py) to run the multi-classification experiments with use_pretrained set to True (only available for 7000-bus case).

If you use the UC_ML models, please cite the following reference using the following entry.

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