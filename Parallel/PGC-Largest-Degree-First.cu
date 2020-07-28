#include <stdio.h>
#include <cuda.h>
#include <bits/stdc++.h>
using namespace std;

void createGraph(int &n, int &m, int &maxDegree, int &minDegree, int** h_degree, int** h_adj, int** h_adj_p)
{
    int ch, i;
    cout << "Enter number of vertices\n";
    cin >> n;
    cout << "Enter number of edges\n";
    cin >> m;
    int k = (n * (n - 1)) / 2;
    if(k < m){
        cout << "Invalid number of edges...assigning graph to be complete graph" << endl;
        m = k;
    }
    cout << "Enter 0 for random and 1 for manual graph" << endl;
    cin >> ch;
    vector<set<int>> adj(n);

    if(ch == 1){       
        for(i = 0;i < m;i++)
        {
            cout << "Enter edge (0 is the first vertex)\n";
            int u,v;
            cin >> u >> v;
            if(adj[u].find(v) != adj[u].end()){
                cout << "Edge already present" << endl;
                i--;
                continue;
            }
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }
    else if(ch == 0){
        i = 0;
        while(i < m){
            int u = rand() % n;
            int v = rand() % n;
            if(adj[u].find(v) != adj[u].end())
                continue;
            if(u == v)
                continue;
            adj[u].insert(v);
            adj[v].insert(u);
            i++;
        }
    }

    *h_adj_p = new int[n + 1];
    *h_adj = new int[(2 * m) + 1];
    *h_degree = new int[n];

    int point = 0;
    for(int i = 0;i < n; i++){
        (*h_adj_p)[i] = point;
        for(auto j : adj[i])
            (*h_adj)[point++] = j;
    }

    (*h_adj_p)[n] = point;
    int mx = INT_MIN, mn = INT_MAX; 
    for(int i = 0;i < n; i++)
    {
        (*h_degree)[i] = (int)adj[i].size();
        mx = max(mx, (int)adj[i].size());
        mn = min(mn, (int)adj[i].size());
    } 
    minDegree = mn;
    maxDegree = mx;
    cout << "Max degree is : " << maxDegree << " and min degree is : " << minDegree << endl;
}

void randomWeightAssign(int* h_weights, int n){                                             //Assign random weights to the vertices
    int i, j = 1;
    vector<int> arr(n);
    for(i = 0; i < n; i++)
        arr[i] = i;
    i = 0;
    while(i < n){
        int x = rand() % arr.size();
        h_weights[arr[x]] = j;
        j++;
        arr.erase(arr.begin() + x);
        i++;
    }
    return;
}

__global__ void initializeRemCol(bool* d_rem, int* d_colours, int n){               //Set uncolored vertex set to true and initial colour to invalid(0)
    int ind = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(ind >= n)
        return;
    d_rem[ind] = true;
    d_colours[ind] = 0;
}

__global__ void colorSet(int* d_adj, int* d_adj_p, int* d_weights, bool* d_rem, int* d_colours, int n, int maxDegree, int* d_degree){//pass h_deg here
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(index >= n)
        return;
    
    if(!d_rem[index])                                                   //If current vertex is already colored, return
        return;

    int i, j, maxColours = maxDegree + 1;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Check if any uncolored neighbour has higher weight        
        j = d_adj[i];                                                                   
        if(d_rem[j] && (d_degree[index]<d_degree[j]  || (d_degree[index]==d_degree[j] && d_weights[j] > d_weights[index])) )  
            return;
    }

    d_rem[index] = false;                                               //Current vertex should be coloured so remove it from remaining set

    bool* forbidden = new bool[maxColours + 1];
    for(i = 0; i < maxColours + 1; i++)
        forbidden[i] = false;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Find out neighbours colors
        j = d_adj[i];
        forbidden[d_colours[j]] = true;
    }
    for(i = 1; i <= maxColours; i++){                                   //Assign color which is not there in neighbours' colors
        if(!forbidden[i]){
            d_colours[index] = i;
            delete [] forbidden;
            return;
        }
    }
    delete [] forbidden;
}

int* colourGraph(int n, int m, int maxDegree, int* d_adj, int* d_adj_p, int* d_weights, int* h_degree){
    int i, rem_size = n, mx = min(n, 512);                                               //Initialize all variables
    bool* h_rem = new bool[n];
    bool* h_inc = new bool[n];
    int* h_colours = new int[n];
    bool* d_rem;
    int *d_colours;
    int *d_degree;

    cudaMalloc((void**)&d_degree, sizeof(int) * n); 
    cudaMalloc((void**)&d_colours, sizeof(int) * n);                                        //Allocate space on GPU
    cudaMalloc((void**)&d_rem, sizeof(bool) * n);

    cudaMemcpy(d_degree, h_degree, sizeof(int) * n, cudaMemcpyHostToDevice);

    initializeRemCol<<<mx, mx>>>(d_rem, d_colours, n);                                       

    while(rem_size > 0){
        // cout << rem_size << endl;
        colorSet<<<mx, mx>>>(d_adj, d_adj_p, d_weights, d_rem, d_colours, n, maxDegree, d_degree);    //Launch kernel
        cudaMemcpy(h_rem, d_rem, sizeof(bool) * n, cudaMemcpyDeviceToHost);                 //Copy back the updated uncolored set
        int k = 0;
        for(i = 0; i < n; i++){
            if(h_rem[i])
                k++;
        }
        rem_size = k;
    }

    cudaMemcpy(h_colours, d_colours, sizeof(int) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_colours);                                                                    //Free memory
    cudaFree(d_rem);
    cudaFree(d_degree);

    return h_colours;
}

int main()
{
    int n=0, m=0, maxDegree=0, minDegree=0;
    int* h_adj = NULL, *h_adj_p = NULL;
    int* d_adj, *d_adj_p;
    int* h_degree = NULL;
    srand(time(0)); 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    createGraph(n, m, maxDegree, minDegree, &h_degree, &h_adj, &h_adj_p);
    
    cudaMalloc((void**)&d_adj, sizeof(int) * ((2 * m) + 1));
    cudaMalloc((void**)&d_adj_p, sizeof(int) * (n + 1));
    
    cudaMemcpy(d_adj, h_adj, sizeof(int) * ((2 * m) + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_p, h_adj_p, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);

    int* h_weights = new int[n];
    int* d_weights;
    cudaMalloc((void**)&d_weights, sizeof(int) * n);
    randomWeightAssign(h_weights, n);
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);

    // for(int i=0;i<n;i++)
    //     cout << h_weights[i] << endl;
    cudaEventRecord(start);

    int *colouring = colourGraph(n, m, maxDegree, d_adj, d_adj_p, d_weights, h_degree);  

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // for(int i=0;i<n;i++)
    //     printf("Vertex %d : %d\n", i, colouring[i]);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n",milliseconds);

    cudaFree(d_adj);
    cudaFree(d_adj_p);
    cudaFree(d_weights);
}

//(1000, 10000) : 9.62ms, 10.49ms, 4.111ms, 4.602ms, 4.441ms
//(1000, 100000) : 74.09ms, 71.66ms, 75.38ms, 76.79ms, 69.658ms
//(10000, 500000) : 82.30ms, 86.9137ms, 82.81ms, 80.86ms, 82.907ms
//(10000, 1000000) : 177.944ms, 175.233ms, 166.64ms, 180.741ms, 180.811ms 
