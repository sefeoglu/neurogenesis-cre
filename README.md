# neurogenesis-cre
Integration of neurogenesis for continual relation extraction utilizing BERT models.
Neurogenesis has been used to provide natural memorization of learnt model with asteroid regularize Q and K.


## Usage and Samples
 * Full Continual Relation Extraction
 * Few Shot Continual Relation Extraction

## Folder Hierarchy
````bash 
.
├── LICENSE
├── README.md
├── data
│   └── README.md
├── docs
│   └── README.md
├── results
│   └── README.md
└── src
    ├── data-preparation
    │   └── README.md
    ├── models
    └── viz
        ├── README.md
        └── plots.py

````

## How it works
```bash
cd src/models/
python train_bert.py
python test_bet.py
```

## Related Works
````
@inproceedings{draelos_2017,
  author={Timothy J. Draelos and Nadine E. Miner and Christopher C. Lamb and Jonathan A. Cox and Craig M. Vineyard and Kristofor D. Carlson and William M. Severa and Conrad D. James and James B. Aimone},
  title={Neurogenesis deep learning: Extending deep networks to accommodate new classes},
  year={2017},
  cdate={1483228800000},
  pages={526-533},
  url={https://doi.org/10.1109/IJCNN.2017.7965898},
  booktitle={IJCNN}
}
````
