#include<stdio.h>
#include<cuda.h>
#include<bits/stdc++.h>
using namespace std;

void readGraph(int &n, int &m, int &maxDegree, int** h_adj, int** h_adj_p){
    int i, k, c;
    cout << "Enter the number of vertices : " << endl;
    cin >> n;
    cout << "Enter the number of edges : " << endl;
    cin >> m;
    int pos = (n * (n - 1)) / 2;
    if(m > pos){
        cout << "Invalid number of edges." << endl;
        return;
    }

    cout << "Random graph or manual? Enter 0 or 1" << endl;
    cin >> c;

    vector<set<int>> adj(n);
    if(c == 0){
        i = 0;
        while(i < m){
            int x, y;
            do{
                x = rand() % n;
                y = rand() % n;
            }while(x == y);

            if(adj[x].find(y) != adj[x].end())
                continue;
            // printf("%d --- %d\n", x, y);
            adj[x].insert(y);
            adj[y].insert(x);
            i++;
        }
    }
    else{
        i = 0;
        while(i < m){
            printf("Click 1 to enter edge and 0 to finish.\n");
            scanf("%d", &k);
            if(!k)
                break;
            int s, d;
            printf("Enter start and end of edge in 1-ordering : \n");
            scanf("%d %d", &s, &d);
            if(s == d){
                printf("Invalid edge.\n");
                continue;
            }
            if(s > n || s < 1 || d > n || d < 1){
                printf("Invalid edge.\n");
                continue;
            }
            adj[s - 1].insert(d - 1);
            adj[d - 1].insert(s - 1);
            i++;
        }
    }
    *h_adj_p = new int[n + 1];
    *h_adj = new int[(2 * m) + 1];

    int point = 0;
    for(i = 0;i < n; i++){
        (*h_adj_p)[i] = point;
        for(auto j : adj[i])
            (*h_adj)[point++] = j;
    }
    (*h_adj_p)[n] = point;

    int mx = INT_MIN; 
    for(i = 0;i < n; i++)
        mx = max(mx, (int)adj[i].size());
    
    maxDegree = mx;
}

__global__ void setKernel(int n, int maxDegree, int* d_adj, int* d_adj_p, int* d_conflicts, int* d_colours){
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(index >= n)
        return;
    if(!d_conflicts[index])
        return;
    int i, j;
    const int maxColours = maxDegree + 1;
    bool* forbidden = new bool[maxColours + 1];
    for(j = 0; j < maxColours + 1; j++)
        forbidden[j] = false;
    
    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){
        int k = d_adj[i];
        forbidden[d_colours[k]] = true;
    }

    for(i = 1; i <= maxColours; i++){
        if(!forbidden[i]){
            d_colours[index] = i;
            return;
        }
    }
    delete[] forbidden;
}

__global__ void checkKernel(int n, int maxDegree, int* d_adj, int* d_adj_p, int* d_conflicts, int* d_colours, bool* d_isConflict){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= n)
        return;

    d_conflicts[index] = false;
    int i;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){
        int k = d_adj[i];
        if(d_colours[k] == d_colours[index] && k < index){
            d_conflicts[index] = true;
            *(d_isConflict) = true;
        }
    }
}

void setColours(int n, int maxDegree, int* d_adj, int* d_adj_p, int* d_conflicts, int* d_colours){
    int mx = max(n, 512);
    setKernel<<<mx, mx>>>(n, maxDegree, d_adj, d_adj_p, d_conflicts, d_colours);
    cudaDeviceSynchronize();
}

bool checkConflicts(int n, int maxDegree, int* d_adj, int* d_adj_p, int* d_conflicts, int* d_colours){
    bool h_isConflict = false;
    bool* d_isConflict;

    cudaMalloc((void**)&d_isConflict, sizeof(bool));
    if(d_isConflict == NULL){
        cout << "Memory full" << endl;
        exit(0);
    }

    cudaMemcpy(d_isConflict, &h_isConflict, sizeof(bool), cudaMemcpyHostToDevice);
    int mx = max(512, n);
    checkKernel<<<mx, mx>>>(n, maxDegree, d_adj, d_adj_p, d_conflicts, d_colours, d_isConflict);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_isConflict, d_isConflict, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_isConflict);

    return h_isConflict;

}

int* colourGraph(int n, int m, int maxDegree, int* d_adj, int* d_adj_p){
    int i;
    bool *h_conflicts = new bool[n]; 
    int *h_colours = new int[n];
    for(i = 0; i < n; i++){
        h_colours[i] = 1;
        h_conflicts[i] = true;
    }

    int* d_conflicts, *d_colours;
    cudaMalloc((void**)&d_conflicts, sizeof(bool) * n);
    cudaMalloc((void**)&d_colours, sizeof(int) * n);

    if(d_colours == NULL){
        cout << "Memory full" << endl;
        exit(0);
    }
    if(d_conflicts == NULL){
        cout << "Memory full" << endl;
        exit(0);
    }

    cudaMemcpy(d_conflicts, h_conflicts, sizeof(bool) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colours, h_colours, sizeof(int) * n, cudaMemcpyHostToDevice);

    do{
        setColours(n, maxDegree, d_adj, d_adj_p, d_conflicts, d_colours);
    }while(checkConflicts(n, maxDegree, d_adj, d_adj_p, d_conflicts, d_colours));

    cudaMemcpy(h_conflicts, d_conflicts, sizeof(bool) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colours, d_colours, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_conflicts);
    cudaFree(d_colours);

    delete[] h_conflicts; 

    return h_colours;
}


int main(){
    int n, m, maxDegree, i;
    srand(time(0)); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* h_adj = NULL, *h_adj_p = NULL;
    int* d_adj, *d_adj_p;

    readGraph(n, m, maxDegree, &h_adj, &h_adj_p);

    cudaMalloc((void**)&d_adj, sizeof(int) * ((2 * m) + 1));
    if(d_adj == NULL){
        cout << "Memory full" << endl;
        exit(0);
    }
    cudaMalloc((void**)&d_adj_p, sizeof(int) * (n + 1));
    if(d_adj_p == NULL){
        cout << "Memory full" << endl;
        exit(0);
    }

    cudaMemcpy(d_adj, h_adj, sizeof(int) * ((2 * m) + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_p, h_adj_p, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);

    cout << "The max degree is : "<< maxDegree << endl;

    cudaEventRecord(start);

    int *colouring = colourGraph(n, m, maxDegree, d_adj, d_adj_p);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    for(i = 0;i < n; i++)
        cout << "Colour of node " << i << " is : " << colouring[i] << endl;
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n",milliseconds);

    cudaFree(d_adj_p);
    cudaFree(d_adj);    
}

//(100, 1000) : 2ms, 1.914ms, 1.88ms, 0.86ms, 1.29ms
