import torch
import torch.nn as nn
from base_network import Linear_layer

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def addColumn(self, col = None, msg = None, out_dim: int = None):
        if not col:
            if self.colGen is None:
                raise ValueError("[Doric]: No column or generator supplied.")
            parents = [colRef for colRef in self.columns]
            col = self.colGen.generateColumn(parents, msg)
        # If caller provided an output dimension, update the final block
        if out_dim is not None:
            # target the last block (final linear layer)
            blk = col.blocks[-1]
            # determine input size
            in_size = getattr(blk, 'inSize', None)
            if in_size is None:
                # try to infer from existing module
                try:
                    in_size = blk.module.linear_layer.in_features
                except Exception:
                    try:
                        in_size = blk.module.in_features
                    except Exception:
                        raise ValueError("Cannot determine block input size to set output dim")

            blk.outSize = out_dim

            args = getattr(blk.module, 'args', None)
            blk.module = Linear_layer(in_size, out_dim, args).to(self.device)
            
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


if __name__ == "__main__":
    ## Example usage of ProgNet with a simple column generator.
    from progDense_block import ProgDenseBlock, ProgLstmBlock1, ProgLstmBlock2
    from progColumn import ProgColumn
    from progNet import ProgNet
    from progBlock_Column_template import ProgColumnGenerator
    
    # Initialize the PNN model
    class PNN_model(ProgColumnGenerator):
        def __init__(self, input_dim, out_dim, hidden_size, lat_connect, device):
            self.ids = 0
            self.input_dim = input_dim
            self.out_dim = out_dim
            self.hidden_size = hidden_size
            self.lat_connect = lat_connect
            self.device = device

        def generateColumn(self, parentCols, msg = None): 
            b1 = ProgLstmBlock1(inSize=self.input_dim, hidden_size=self.hidden_size, numLaterals=0, lat_connect=self.lat_connect)
            b2 = ProgLstmBlock2(self.hidden_size, self.hidden_size, len(parentCols), lat_connect=self.lat_connect, args=None)
            b3 = ProgDenseBlock(self.hidden_size, self.out_dim, len(parentCols), activation = None, args=None)
            c = ProgColumn(self.__genID(), [b1, b2, b3], device = self.device, parentCols = parentCols)
            return c

        def __genID(self):
            id = self.ids
            self.ids += 1
            return id
        
    model = ProgNet(colGen = PNN_model(input_dim=6, out_dim=4, hidden_size=64, lat_connect="all", device="cuda"))
    model.addColumn(out_dim=10)
    x = torch.randn((2, 6), device="cuda"). unsqueeze(1)
    print("x shape ", x.shape)
    print(model(x, 0, 0, 0))