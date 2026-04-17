import torch
import torch.nn as nn

"""
Class that acts as the base building-blocks of ProgNets.
Includes a module (usually a single layer),
a set of lateral modules, and an activation.
"""
class ProgBlock(nn.Module):
    """
    Runs the block on input x.
    Returns output tensor or list of output tensors.
    """
    def runBlock(self, x):
        raise NotImplementedError

    """
    Runs lateral i on input x.
    Returns output tensor or list of output tensors.
    """
    def runLateral(self, i, x):
        raise NotImplementedError

    """
    Runs activation of the block on x.
    Returns output tensor or list of output tensors.
    """
    def runActivation(self, x):
        raise NotImplementedError

    """
    Returns a dictionary of data about the block.
    """
    def getData(self):
        raise NotImplementedError

    """
    Returns True if block is meant to contain laterals.
    Returns False if block is meant to be a utility with not lateral inputs.
    Default is True.
    """
    def isLateralized(self):
        return True
    

"""
Conveniance class for un-lateralized blocks.
"""
class ProgInertBlock(ProgBlock):
    def isLateralized(self):
        return False


"""
Class that generates new ProgColumns using the method generateColumn.
The parentCols list will contain references to each parent column,
such that columns can access lateral outputs.
Additional information may be passed through the msg argument in
generateColumn and ProgNet.addColumn.
"""
class ProgColumnGenerator:
    def generateColumn(self, parentCols, msg = None):
        raise NotImplementedError