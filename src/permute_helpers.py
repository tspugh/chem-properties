import torch
import itertools

#https://github.com/mcunow/graph-matching/blob/main/src/matching.py#L26

def permute_graphs_over_n(n):
    permutations = list(itertools.permutations(range(n)))
    ls=[]
    dic={}
    k=n
    for i in range(n):
        for j in range(i,n):
            if i != j:  # Avoid self-loops
                dic[(i,j)]=k
                dic[(j,i)]=k
                k += 1
    for perm in permutations:
        temp=permute_graph(perm,dic)
        ls.append(temp)
    return torch.tensor(ls)

def permute_graph(perm,dic):
    ls={}
    for idx,p in enumerate(perm):
        ls[idx]=p

    for i in range(len(perm)):
        for j in range(i,len(perm)):
            if i!=j:
                temp=dic[(i,j)]
                key=((ls[i],ls[j]))
                ls[temp]=dic[key]
    return list(ls.values())

def create_permutations():
    ls=[]
    for i in range (1,10):
        ls.append(permute_graphs_over_n(i))
    return ls