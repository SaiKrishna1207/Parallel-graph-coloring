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

__global__ void colorKernel(int n, int weight, int maxColors, int* d_adj, int* d_adj_p, int* d_weights, int* d_colors, bool* d_sameWt)
{
    int index = (blockDim.x * blockIdx.x) + threadIdx.x;    
    if(index >= n || d_colors[index]!=-1 || d_weights[index]!=weight)
        return;
    
    //tried atomically changing color and then checking for conflicts with neighbors with the same weight
    //but it gave the same color for all the vertices lmao
    //if we do it with that detect conflicts do-while loop from naive implementation, it will lead to no significant gain in time complexity
    
        

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
        cout << "Weight of " << i << "is " << h_weights[i] << endl;
        max_wt = max(max_wt, h_weights[i]);
    }
        
    cout << max_wt << endl;
    
    int* h_colors = new int[n];
    memset(h_colors,0,sizeof(int)*n);
    bool* h_sameWt = new bool[n];
    memset(h_sameWt, false, sizeof(bool)*n);

    int* d_colors, *d_weights;
    bool* d_sameWt;
    
    cudaMalloc((void**)&d_weights, sizeof(int) * n);
    cudaMalloc((void**)&d_colors, sizeof(int) * n);
    cudaMalloc((void**)&d_sameWt, sizeof(bool) * n);
    
    cudaMemcpy(d_weights, h_weights, sizeof(int) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_colors, h_colors, sizeof(int) * n, cudaMemcpyHostToDevice);
   
    int mx = min(n,512);
    for(int i = max_wt; i>=1; i--)
    {
        cudaMemcpy(d_sameWt, h_sameWt, sizeof(bool) * n, cudaMemcpyHostToDevice);
        colorKernel<<<mx,mx>>>(n, i, maxDegree, d_adj, d_adj_p, d_weights, d_colors, d_sameWt);
    }
    cudaMemcpy(h_colors, d_colors, sizeof(int) * n, cudaMemcpyDeviceToHost);
    // for(int i=0;i<n;i++)
    //     cout << "Color of " << i << "is" << h_colors[i] << endl;

    cudaFree(d_adj);
    cudaFree(d_adj_p);
    cudaFree(d_colors);
    cudaFree(d_sameWt);
    cudaFree(d_weights);
}
