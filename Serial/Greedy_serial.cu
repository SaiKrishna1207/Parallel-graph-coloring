#include<stdio.h>
#include<bits/stdc++.h>
#include <stdlib.h> 
#include<time.h>
#include<cuda.h>

using namespace std;

void setColours(vector<set<int>>& adj, int maxDegree, int n, int m, vector<int> &colours, vector<bool> & conflicts){
    int maxColours = maxDegree + 1;

    int i, j;
    for(i = 0; i < n; i++){
        if(!conflicts[i])
            continue;
        vector<bool> no(maxColours + 1, false);
        for(auto j : adj[i])
            no[colours[j]] = true;
        for(j = 1; j <= maxColours; j++){
            if(!no[j]){
                colours[i] = j;
                break;
            }
        }
    }
}

bool checkConflicts(vector<set<int>>& adj, int maxDegree, int n, int m, vector<int> &colours, vector<bool>& conflicts){
    bool isConflict = false;
    int i;
    for(i = 0; i < n; i++){
        conflicts[i] = false;
        for(auto j : adj[i]){
            if(colours[j] == colours[i] && j < i){
                conflicts[i] = true;
                isConflict = true;
            }
        }
    }
    return isConflict;
}

int main(){
    int i, n, m;
    srand(time(0)); 
    printf("Enter the number of vertices and edges for the graph : \n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    scanf("%d %d", &n, &m);
    vector<set<int>> adj(n);
    for(i = 0; i < m; i++){
        int x, y;
        do{
            x = rand() % n;
            y = rand() % n;
        }while(x == y);
        if(adj[x].find(y) != adj[x].end()){
            i--;
            continue;
        }
        // printf("Edge %d : %d --- %d\n", i + 1, x, y);
        adj[x].insert(y);
        adj[y].insert(x);
    }

    vector<bool> conflicts(n, true);
    vector<int> colours(n, 0);

    int maxDegree = INT_MIN;
    for(i = 0;i < n; i++)
        maxDegree = max(maxDegree, (int)adj[i].size());
    cudaEventRecord(start);
    
    do{
        setColours(adj, maxDegree, n, m, colours, conflicts);
    }while(checkConflicts(adj, maxDegree, n, m, colours, conflicts));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    for(i = 0;i < n; i++)
        printf("Vertex %d --> Colour %d. \n", i, colours[i]);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n",milliseconds);

    return 0;
}

//(100, 1000) : 0.7367ms, 0.7420ms, 0.4576ms, 0.779ms, 0.3501ms
//(1000, 100000) : 64.093ms , 43.645ms, 43.71ms, 43.92ms, 42.958ms
//(10000, 1000000) : 750ms, 645.075ms, 637.407, 645.593ms, 635.579ms
//(10000, 500000) : 313ms, 318.63ms, 316.035ms, 312.555ms, 313.104ms
