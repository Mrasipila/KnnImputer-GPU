import argparse
import pandas 
import cupy as cp

parser = argparse.ArgumentParser(description='Split Coco annotation file and image file into training and test set') 
parser.add_argument('-file',type=str,required=True,dest="file", help="Name of the file to be imputed")
parser.add_argument('-n_neigh',type=str,required=False,dest="n_neigh", help="Number of neighbors in Knn")

args = parser.parse_args()


def KnnImputer(df,n_neigh=5):
    blockSize = 256
    gridSize = int((len(df)+blockSize-1)/blockSize)
    dist_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void knn_impute(float* dataset, float* distances, int num_feat, int *indices, int n_neigh) {
        int gti = blockDim.x * blockIdx.x + threadIdx.x;
        for(int k = 0 ; k < num_feat; k++){ 
            if ((dataset[gti * num_feat + k]!=dataset[gti * num_feat + k])){ // check if cell is nan or not
                int current_nan_cell = gti * num_feat + k;
                int current_nan_col = k;
                float dist = 0;
                for (int i = 0 ; i < num_feat ; i++) {
                    if((dataset[gti * num_feat + i]!=dataset[gti * num_feat + i]) || (dataset[current_nan_cell] != dataset[current_nan_cell])){
                        continue;
                    } else { 
                        dist = dataset[gti * num_feat + i] - dataset[current_nan_cell];
                        dist *= dist;
                    }
                }
                distances[gti] = sqrtf(dist);
                indices[gti] = gti;
                for (int j = 0; j < num_feat; j++) {
                    for (int l = j + 1; l < num_feat; l++) {
                        if (distances[j] > distances[l]) {
                            // Swap distances
                            float tempDist = distances[j];
                            distances[j] = distances[l];
                            distances[l] = tempDist;

                            // Swap indices
                            int tempIdx = indices[j];
                            indices[j] = indices[l];
                            indices[l] = tempIdx;
                        }
                    }
                }
                // Calculate mean of nearest neighbors
                float mean = 0.0f;
                int count = 0;
                for (int j = 0; j < n_neigh; j++) {
                    if (!isnan(dataset[indices[j] * num_feat + current_nan_col])) {
                        mean += dataset[indices[j] * num_feat + current_nan_col];
                        count++;
                    }
                }

                if (count > 0) {
                    mean /= count;
                    dataset[current_nan_cell] = mean;
                } else {
                    // If no valid neighbors found, handle the case (e.g., leave NaN or set to a default value)
                }
            }
        }
    }
    ''', 'knn_impute')
    dataset = cp.asarray(df.to_numpy().flatten()) # we use 1D array to facilitate the prralelization
    distances = cp.array(([99999]*len(df)), dtype=cp.float32) # we setup an array of inf so that the distance of nan feature is not zero, zero means it will be first we will pick the closest points to query
    indices = cp.zeros(len(df), dtype=cp.float32)
    dist_kernel((blockSize,),(gridSize,),(dataset,distances,len(df.iloc[0]),indices,n_neigh))
    return dataset

df = pd.read_csv(str(args.file))
df.dropna(axis=0, how='all')
if args.n_neigh:
    df1 = KnnImputer(df,args.n_neigh)
else:
    df1 = KnnImputer(df)

df = pd.DataFrame(df1.reshape(df.shape[0],df.shape[1]))
df = df.dropna() # drop the row for which there is a cell with no valid neighbors found.
df.to_csv('result.csv')