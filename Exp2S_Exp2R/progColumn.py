import torch
import numpy as np
import torch.nn as nn

"""
A column representing one sequential ANN with all of its lateral modules.
Outputs of the last forward run are stored for child column laterals.
Output of each layer is calculated as:
y = activation(block(x) + sum(laterals(x)))

colID -- A unique identifier for the column.
blockList -- A list of ProgBlocks that will be run sequentially.
parentCols -- A list of pointers to columns that will be laterally connectected.
              If the list is empty, the column is unlateralized.
"""
class ProgColumn(nn.Module):
    def __init__(self, colID, blockList, device, parentCols = []):
        super().__init__()
        self.colID = colID
        self.isFrozen = False
        self.parentCols = parentCols
        self.blocks = nn.ModuleList(blockList).to(device)
        self.numRows = len(blockList)
        self.lastOutputList = []

    def freeze(self, unfreeze = False):
        if not unfreeze:    # Freeze params.
            self.isFrozen = True
            for param in self.parameters():   param.requires_grad = False
        else:               # Unfreeze params.
            self.isFrozen = False
            for param in self.parameters():   param.requires_grad = True

    def getData(self):
        data = dict()
        data["colID"] = self.colID
        data["rows"] = self.numRows
        data["frozen"] = self.isFrozen
        #data["last_outputs"] = self.lastOutputList
        data["blocks"] = [block.getData() for block in self.blocks]
        data["parent_cols"] = [col.colID for col in self.parentCols]
        return data

    def forward(self, input, h0, c0):
        outputs = []
        x = input
        for r, block in enumerate(self.blocks):
            #if isinstance(block, ProgMultiBlock):
                #y = self.__forwardMulti(x, r, block)
            #else:
            y, h0, c0 = self.__forwardSimple(x, h0, c0, r, block)
            outputs.append(y)
            x = y
        self.lastOutputList = outputs
        return outputs[-1], h0, c0

    def __forwardSimple(self, x, h0, c0, row, block):
        currOutput, h0, c0 = block.runBlock(x, h0, c0)
        if not block.isLateralized() or row == 0 or len(self.parentCols) < 1:
            y = block.runActivation(currOutput)
        elif isinstance(currOutput, list):
            for c, col in enumerate(self.parentCols):

                lats = block.runLateral(c, col.lastOutputList[row - 1])
                for i in range(len(currOutput)):
                    if currOutput[i] is not None and lats[i] is not None:
                        currOutput[i] += lats[i]
            y = block.runActivation(currOutput)
        else:
            for c, col in enumerate(self.parentCols):
                currOutput = currOutput.clone() + block.runLateral(c, col.lastOutputList[row - 1])
            y = block.runActivation(currOutput)
        return y, h0, c0

    def __forwardMulti(self, x, row, block):
        if not isinstance(x, list):
            raise ValueError("[Doric]: Multiblock input must be a python list of inputs.")
        currOutput = block.runBlock(x)
        if not block.isLateralized() or row == 0 or len(self.parentCols) < 1:
            y = block.runActivation(currOutput)
        else:
            for c, col in enumerate(self.parentCols):
                lats = block.runLateral(c, col.lastOutputList[row - 1])
                for i, p in enumerate(block.getPassDescriptor()):
                    if not p:   currOutput[i] += lats[i]
            y = block.runActivation(currOutput)
        return y
