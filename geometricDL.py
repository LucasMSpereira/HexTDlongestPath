#%% Imports
import os
os.environ['DGLBACKEND'] = 'pytorch'
import dglUtils
import random
import data_utils
import torch
import dgl
import torch.nn.functional as F
random.seed(100)
#%% Data
rowAmount = colAmount = 10 # map dimensions
nNodes = rowAmount * colAmount
ds = data_utils.dataManager(0, rowAmount, colAmount)
bSize = 100
trainBatchLoader, valBatchLoader = ds.readDGLdataset(
  trainPercent = 0.05, batchSize = bSize
)
ba = next(iter(trainBatchLoader))
print(ba)
#%% Instantiate model with parameters
batchNumNodes = nNodes * bSize # num of nodes in batched graph
model = dglUtils.gcn({
  "inDim": batchNumNodes,
  "hDim": batchNumNodes,
  "activFunction": F.hardswish
})
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
lossHist = []
for epoch in range(2):
  for batchID, (batched_graph, OSPlength) in enumerate(trainBatchLoader):
    pred = model(batched_graph, batched_graph.ndata["initialType"].float())
    loss = F.mse_loss(pred.float(), OSPlength.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lossHist.append(loss)
    print(batchID + 1, f"{lossHist[-1].item():.3e}")