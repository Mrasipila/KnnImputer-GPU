Simple tool That implement KNN Imputation for GPUs

## Dependencies

- Cupy
- Pandas
- argparse

## How it works

```
python KnnImputer.py -h
usage: KnnImputer.py [-h] -file FILE [-n_neigh N_NEIGH]

Split Coco annotation file and image file into training and test set

optional arguments:
  -h, --help        show this help message and exit
  -file FILE        Name of the file to be imputed
  -n_neigh N_NEIGH  Number of neighbors in Knn

```

## Example running 

``` 
python KnnImputer.py -file "dataframe.csv" -n_neigh 5
```
