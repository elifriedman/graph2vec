# graph2vec v2

graph2vec is to node2vec as doc2vec is to word2vec

node2vec performs random walks on a graph in order to create sequences that can be fed into word2vec
graph2vec performs random walks on multiple graphs and feeds those walks, along with a unique graph ID per graph into doc2vec, which learns a continuous embedding for each node and each graph.

Note that this technique is the same as [this graph2vec](https://arxiv.org/abs/1707.05005) but they use a more complicated "word" equivalent--rooted subgraphs around each node--rather than the random walks that are performed here. And the results here are better than the ones reported there.

### Basic Usage

#### Example
First preprocess the [datasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets):
```
python src/process_graph.py graph/PROTEINS
```
This separates the graphs into individual gpickled networkx files located in the DATASET/graphs/ folder

Then run graph2vec to extract the embeddings:
```
python src/main.py --p 1 --q 0.5 --dimensions 1024 --input graph/PROTEINS/graphs --output emb/PROTEINS_p1_q.5_d1024.emb
```
This will save the embedding

Then run the classification
```
python src/classify.py graph/PROTEINS
```
which will output the accuracy on a random test set of 10% of the data.
An SVM is used to train the data, with parameter tuning done using 5-fold cross validation on the train set.

#### Results
| Dataset | Score |
| ------- | ----- |
| PROTEINS   | 0.836 (+/- 0.06) |
| MUTAG   | 0.842 (+/- 0.18) |
| NCI1    | 0.808 (+/- 0.03) |
| NCI109    | 0.785 (+/- 0.04) |
| PTC     | 0.697 (+/- 0.12) |


