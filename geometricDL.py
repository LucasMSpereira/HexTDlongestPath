#%% Imports
import os
os.environ['DGLBACKEND'] = 'pytorch'
from datetime import datetime
import matplotlib
import dglUtils
import random
import data_utils
from pandas import DataFrame as df
import torch
import seaborn as sns
import torch.nn.functional as F
random.seed(100)
#%% Data
rowAmount = colAmount = 10 # map dimensions
nNodes = rowAmount * colAmount
ds = data_utils.dataManager(0, rowAmount, colAmount)
bSize = 100
trainBatchLoader, valBatchLoader = ds.readDGLdataset(
  trainPercent = 0.7, batchSize = bSize
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
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
lossHist = df(columns = ['Batch', 'MSE loss'])
dfPos = -1
for epoch in range(2):
  print(f"""Epoch {epoch + 1} {datetime.now().strftime("%H:%M:%S")}""")
  for batchID, (batched_graph, OSPlength) in enumerate(trainBatchLoader):
    dfPos += 1
    pred = model(batched_graph, batched_graph.ndata["initialType"].float())
    loss = F.mse_loss(pred.float(), OSPlength.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lossHist.loc[dfPos] = [batchID + 1, loss.item()]
    if random.random() > 0.95:
      print(f"Batch: {batchID + 1}   Loss: {lossHist.iloc[-1]['MSE loss']:.3e}")
  facetgrid = sns.relplot(data = lossHist, kind = 'line', x = 'Batch', y = 'MSE loss')
  facetgrid.ax.set_yscale("log")
  matplotlib.pyplot.show()