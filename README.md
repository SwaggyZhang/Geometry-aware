# Geometry-aware


## Training Details

We spilt the whole dataset into base classes and novel classes. And the base class dataset is 
divided into **base_train**, **base_val** and **base_test** subsets.

Base_train, base_val and base_test have the same categories. The samples in **base_val** are
used for incremental sessions.
### The first session
We use the **base_train** file to construct the first session. In order to guarantee the model having
the ability to deal with tasks in few-shot scenarios, we utilize the episodic paradigm to train
the model.

The model in first-session stage consists of BERT and vanilla prototypical networks. After being trained with
a series of N-way K-shot tasks, all samples of each category are fed into BERT model and prototypical network
to get the embeddings and the prototypes. 

We choose the sample which has the shortest distance to the prototype of current category as an exemplar.
And we leverage the exemplars to construct the base graph.

### Incremental sessions
