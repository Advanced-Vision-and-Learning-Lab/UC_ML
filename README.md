# Unit Commitment Machine Learning:
**A Fast Learning-Based Unit Commitment Strategy with AC Optimal Power Flow for Large Grids with Direct Inclusion of Weather**

Farnaz Safdarian, Joshua Peeples, Dillon Richards, Jessica Wert, Thomas Overbye

Note: If this code is used, cite it: Farnaz Safdarian, Joshua Peeples, Dillon Richards, Jessica Wert, Thomas Overbye. (2023, November 9) Peeples-Lab/UCML: Initial Release (Version v1.0). 

[`Zendo`](https://doi.org/10.5281/zenodo.10092401)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10092401.svg)](https://doi.org/10.5281/zenodo.10092401)

[`arXiv`]()

[`BibTeX`](#Citing)

In this repository, we provide the paper and code for "A Fast Learning-Based Unit Commitment Strategy with AC Optimal Power Flow for Large Grids with Direct Inclusion of Weather."

## Installation Prerequisites


The [`requirements.txt`](requirements.txt) file includes all the necessary packages, and the packages will be installed using:

   ```pip install -r requirements.txt```

## Demo

To get started, please follow the instructions in the [Datasets](Datasets) and [Pretrained_Models](Pretrained_Models) folders to download the dataset(s) and model(s) respectively.
Next, run [`demo.py`](demo.py) in Python IDE (e.g., Spyder) or command line to train, validate, and test models. 

## Main Functions

The UC ML code uses the following functions. 

1. Load data  

   ```features, labels = load_dataset(**Parameters)```

2. Prepare dataset(s) for model (train/validation/test split)
   
   ``` X_train, X_test, y_train, y_test = train_test_split(**Parameters)```

3. Cross validate model(s) for hyperparameter tuning 

   ```classifiers = cross_validation(**Parameters)```

4. Retrain best model(s) with full training data, test model, and get quantitative and qualitative results

   ```generate_PR_Curves(**Parameters)```

## Parameters

The parameters can be set on the command line and the parameters for the experiments are stored in the [`Demo_Parameters.py`](Demo_Parameters.py).

## Inventory

```
https://github.com/Advanced-Vision-and-Learning-Lab/UC_ML

└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameter file for the demo.
    └── Utils  //utility functions
        ├── crossval.py  // Contains functions perform hyperparameter tuning for each classifier. 
        ├── PR_curves.py  // Generate precision-recall curves for each classifier.
     
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2023 F. Safdarian, J. Peeples, D. Richards, J. Wert, T. Overbye. All rights reserved.

## <a name="Citing"></a>Citing Unit Commitment Machine Learning

If you use the Unit Commitment Machine Learning code, please cite the following reference using the following entry.

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