import argparse
import pandas 
import cupy as cp

parser = argparse.ArgumentParser(description='Knn Imputer for GPU') 
parser.add_argument('-file',type=str,required=True,dest="file", help="Name of the file to be imputed")
parser.add_argument('-n_neigh',type=str,required=False,dest="n_neigh", help="Number of neighbors in Knn")

args = parser.parse_args()


def KnnImputer(df,n_neigh=5):
    blockSize = 256
    gridSize = int((len(df)+blockSize-1)/blockSize)
    dist_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void knn_impute(float* dataset, float* distances, int num_feat, int* indices, int num_rows, int n_neigh, int** nan_col,int num_nan) {
        int gti = blockIdx.x * blockDim.x + threadIdx.x;

        if (gti >= num_rows) return;
            if (isnan(dataset[gti * num_feat + k])) { // check if cell is NaN

                // Initialize distances and indices
                for (int i = 0; i < num_rows; i++) {
                    indices[i] = i;
                    distances[i] = 99999.0f; // Initialize with a large value
                }

                // Compute distances to other rows
                 int cpt = 0;
                for (int k = 0; k < num_rows; k++) {
                    if (i == gti) continue; // Skip self
                    float dist = 0.0f;
                    bool has_valid_data = false;
                    for (int j = 0; j < num_feat; j++) {
                        if (j == k) continue; // Skip the NaN feature itself
                        float val1 = dataset[gti * num_feat + j];
                        float val2 = dataset[k * num_feat + j];
                        if (!isnan(val1) && !isnan(val2)) {
                            float diff = val1 - val2;
                            dist += diff * diff;
                            has_valid_data = true;
                        }
                        if (isnan(val2)) {
                            nan_col[cpt][0] = j;
                            nan_col[cpt][1] = k;
                            cpt++;
                        }
                    }
                    if (has_valid_data) {
                        distances[i] = sqrtf(dist);
                    }
                }

                // Sort the distances and indices based on distances
                for (int i = 0; i < num_rows - 1; i++) {
                    for (int j = i + 1; j < num_rows; j++) {
                        if (distances[j] < distances[i]) {
                            float tempDist = distances[i];
                            distances[i] = distances[j];
                            distances[j] = tempDist;

                            int tempIdx = indices[i];
                            indices[i] = indices[j];
                            indices[j] = tempIdx;
                        }
                    }
                }

                // Calculate the mean of the nearest neighbors for imputation
                float mean = 0.0f;
                int count = 0;
                for (int i = 0; i < num_rows; i++) {
                    for (int j = 0; j < n_neigh; j++) {
                        int neighbor_idx = indices[i];
                        if (neighbor_idx == gti) continue; // Skip the NaN row itself
                        float neighbor_val = dataset[neighbor_idx * num_feat + k];
                        if (!isnan(neighbor_val)) {
                            mean += neighbor_val;
                            count++;
                        }
                    }
                }
                int ty = threadIdx.y;
                if (ty < n_neigh){
                    for (int i=0; i < num_nan; i++) {
                        indices[ty]
                    }
                }

                if (count > 0) {
                    mean /= count;
                    dataset[gti * num_feat + k] = mean; // Impute the NaN value
                } else {
                    dataset[gti * num_feat + k] = 0.0f; // Set to zero as default if no valid neighbors found
                }
            }
        }
    }
    ''', 'knn_impute')
    dataset = cp.asarray(df.to_numpy().flatten()) # we use 1D array to facilitate the prralelization
    distances = cp.array(([99999]*len(df)), dtype=cp.float32) # we setup an array of inf so that the distance of nan feature is not zero, zero means it will be first we will pick the closest points to query
    indices = cp.zeros(len(df), dtype=cp.float32)
    nan_col = cp.array(([0,0]*cp.count_nonzero(cp.isnan(dataset)))),dtype=cp.int32)
    dist_kernel((blockSize,),(gridSize,),(dataset,distances,len(df.iloc[0]),indices,n_neigh,nan_col,cp.count_nonzero(cp.isnan(dataset))))
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
