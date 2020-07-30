#include <stdio.h>
#include <cuda.h>
#include <bits/stdc++.h>
using namespace std;

void createGraph(int &n, int &m, int &maxDegree, int &minDegree, int** h_degree, int** h_adj, int** h_adj_p)
{
    int i, ch;
    cout << "Enter number of vertices\n";
    cin >> n;
    cout << "Enter number of edges\n";
    cin >> m;
    int k = ((n * (n - 1)) / 2);
    if(k < m){
        cout << "Too many edges...making it complete graph." << endl;
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
    cout << "Max degree is : " << mx << " and Min degree is : " << mn << endl;
}

__global__ void assignWeightKernel(int n, int k, int weight, int* d_adj, int* d_adj_p, bool* d_visited, int* d_weights)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    if( index >= n || d_visited[index] )
        return;
    
    int deg = 0;
    int start = d_adj_p[index];
    int end = d_adj_p[index + 1];
    for(int i = start; i < end; i++)
    {
        int neighbor = d_adj[i];
        if( !d_visited[neighbor] )
            deg++;
    }
    __syncthreads();

    if(deg <= k)
    {
        d_visited[index] = true;
        d_weights[index] = weight;
    }
}

void randomWeightAssign(int* h_rweights, int n){                                             //Assign random weights to the vertices
    int i, j = 1;
    vector<int> arr(n);
    for(i = 0; i < n; i++)
        arr[i] = i;
    i = 0;
    while(i < n){
        int x = rand() % arr.size();
        h_rweights[arr[x]] = j;
        j++;
        arr.erase(arr.begin() + x);
        i++;
    }
    return;
}

void assignWeight(int n, int* d_adj, int* d_adj_p, int* h_weights, bool* h_visited)
{
    int* d_weights=NULL;
    bool* d_visited=NULL;

    cudaMalloc((void**)&d_weights, sizeof(int) * n);
    cudaMalloc((void**)&d_visited, sizeof(int) * n);
    
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, sizeof(bool) * n, cudaMemcpyHostToDevice);

    int k = 1;
    int weight = 1;
    int cnt = 1;
    int mx = min(n, 512);

    while(cnt != 0)
    {
        assignWeightKernel<<<mx,mx>>>(n, k, weight, d_adj, d_adj_p, d_visited, d_weights);
        cudaDeviceSynchronize();
        cudaMemcpy(h_visited, d_visited, sizeof(bool) * n, cudaMemcpyDeviceToHost);

        cnt = 0;
        for(int i = 0;i < n; i++)
        {
            if(!h_visited[i])
                cnt++;
        }
        k++;
        weight++;
    }
    cudaMemcpy(h_weights, d_weights, sizeof(int) * n, cudaMemcpyDeviceToHost);
}

__global__ void initializeRemCol(bool* d_rem, int* d_colours, int n){               //Set uncolored vertex set to true and initial colour to invalid(0)
    int ind = (blockDim.x * blockIdx.x) + threadIdx.x;
    if(ind >= n)
        return;
    d_rem[ind] = true;
    d_colours[ind] = 0;
} 

__global__ void colorSet(int* d_adj, int* d_adj_p, int* d_weights, bool* d_rem, int* d_colours, int n, int maxDegree){//pass h_deg here
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(index >= n)
        return;
    
    if(!d_rem[index])                                                   //If current vertex is already colored, return
        return;

    int i, j, maxColours = maxDegree + 1;

    for(i = d_adj_p[index]; i < d_adj_p[index + 1]; i++){               //Check if any uncolored neighbour has higher weight or if equal, higher index       
        j = d_adj[i];                                                                   
        if(d_rem[j] && ((d_weights[j] > d_weights[index]) || (d_weights[j] == d_weights[index] && j>index)) )  
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

int* colourGraph(int n, int m, int maxDegree, int* d_adj, int* d_adj_p, int* d_weights){
    int i, rem_size = n, mx = min(n, 512);                                               //Initialize all variables
    bool* h_rem = new bool[n];
    bool* h_inc = new bool[n];
    int* h_colours = new int[n];
    bool* d_rem;
    int *d_colours;

    cudaMalloc((void**)&d_colours, sizeof(int) * n);                                        //Allocate space on GPU
    cudaMalloc((void**)&d_rem, sizeof(bool) * n);

    initializeRemCol<<<mx, mx>>>(d_rem, d_colours, n);                                       

    while(rem_size > 0){
        // cout << rem_size << endl;
        colorSet<<<mx, mx>>>(d_adj, d_adj_p, d_weights, d_rem, d_colours, n, maxDegree);    //Launch kernel
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

    return h_colours;
}

int main()
{
    int n=0, m=0, maxDegree=0, minDegree=0;
    int* h_adj = NULL, *h_adj_p = NULL;
    int* d_adj, *d_adj_p;
    int* h_degree = NULL;
    
    createGraph(n, m, maxDegree, minDegree, &h_degree, &h_adj, &h_adj_p);
    
    cudaMalloc((void**)&d_adj, sizeof(int) * ((2 * m) + 1));
    cudaMalloc((void**)&d_adj_p, sizeof(int) * (n + 1));
    
    cudaMemcpy(d_adj, h_adj, sizeof(int) * ((2 * m) + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_p, h_adj_p, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);

    int* h_weights = new int[n];    
    bool* h_visited = new bool[n];
    
    memset(h_weights, -1, n*sizeof(int));
    memset(h_visited, false, n*sizeof(bool));
    
    assignWeight(n, d_adj, d_adj_p, h_weights, h_visited);
    
    int max_wt=INT_MIN;
    for(int i=0;i<n;i++)
    {
        cout << "Weight of " << i << " is " << h_weights[i] << endl;
        max_wt = max(max_wt, h_weights[i]);
    }
        
    cout << max_wt << endl;
    
    int* h_colors = new int[n];
    memset(h_colors,0,sizeof(int)*n);

    int* d_colors, *d_weights;
    
    cudaMalloc((void**)&d_weights, sizeof(int) * n);
    cudaMalloc((void**)&d_colors, sizeof(int) * n);
    
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, h_colors, sizeof(int) * n, cudaMemcpyHostToDevice);
   
    // int* h_rweights = new int[n];
    // int* d_rweights;
    // cudaMalloc((void**)&d_rweights, sizeof(int) * n);
    // randomWeightAssign(h_rweights, n);
    // cudaMemcpy(d_rweights, h_rweights, sizeof(int) * n, cudaMemcpyHostToDevice);

    // for(int i=0;i<n;i++)
    //     cout << i << " " << h_rweights[i] << endl;

    h_colors = colourGraph(n, m, maxDegree, d_adj, d_adj_p, d_weights);

    for(int i=0;i<n;i++)
        cout << "Color of vertex " << i << " is " << h_colors[i] << endl;

    cudaFree(d_adj);
    cudaFree(d_adj_p);
    cudaFree(d_colors);
    cudaFree(d_weights);
    // cudaFree(d_rweights);
}
