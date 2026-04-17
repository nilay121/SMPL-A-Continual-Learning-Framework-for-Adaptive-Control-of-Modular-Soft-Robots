import torch
import torch.nn as nn

"""
A progressive neural network as described in Progressive Neural Networks (Rusu et al.).
Columns can be added manually or with a ProgColumnGenerator.
https://arxiv.org/abs/1606.04671
"""

class ProgNet(nn.Module):
    def __init__(self, colGen = None):
        super().__init__()
        self.columns = nn.ModuleList()
        self.numRows = None
        self.numCols = 0
        self.colMap = dict()
        self.colGen = colGen

    def addColumn(self, col = None, msg = None):
        if not col:
            if self.colGen is None:
                raise ValueError("[Doric]: No column or generator supplied.")
            parents = [colRef for colRef in self.columns]
            col = self.colGen.generateColumn(parents, msg)
        self.columns.append(col)
        if col.colID in self.colMap:
            raise ValueError("[Doric]: Column ID must be unique.")
        self.colMap[col.colID] = self.numCols
        if self.numRows is None:
            self.numRows = col.numRows
        else:
            if self.numRows != col.numRows:
                raise ValueError("[Doric]: Each column must have equal number of rows.")
        self.numCols += 1
        return col.colID

    def freezeColumn(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        col.freeze()

    def freezeAllColumns(self):
        for col in self.columns:
            col.freeze()

    def unfreezeColumn(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        col.freeze(unfreeze = True)

    def unfreezeAllColumns(self):
        for col in self.columns:
            col.freeze(unfreeze = True)

    def isColumnFrozen(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        return col.isFrozen

    def getColumn(self, id):
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        col = self.columns[self.colMap[id]]
        return col

    def forward(self, x, h0, c0, id):
        if self.numCols <= 0:
            raise ValueError("[Doric]: ProgNet cannot be run without at least one column.")
        if id not in self.colMap:
            raise ValueError("[Doric]: No column with ID %s found." % str(id))
        colToOutput = self.colMap[id]
        for i, col in enumerate(self.columns):
            y, h0, c0 = col(x, h0, c0)
            if i == colToOutput:
                return y, h0, c0

    def getData(self):
        data = dict()
        data["cols"] = [c.getData() for c in self.columns]
        return data
