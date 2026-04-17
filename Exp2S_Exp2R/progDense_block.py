import torch
import torch.nn as nn
from progBlock_Column_template import ProgBlock
from base_network import LSTM1, LSTM2, Linear_layer, LSTM_lateral
"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.
"""
def identity(x):
    return x

class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), skipConn = False, lambdaSkip = identity, args=None):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.module = Linear_layer(inSize, outSize, args)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        if activation is None:   
            self.activation = identity
        else:                    
            self.activation = activation

    def runBlock(self, x, hn, cn):
        if self.skipConn:
            self.skipVar = x
        return self.module(x, hn, cn)

    def runLateral(self, i, x):
        print("forwrad lateral called!!")
        x = x.reshape(x.shape[0], -1)
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Dense"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)
    def isLateralized(self):
        return False


"""
A ProgBlock containing a single fully connected layer (nn.Linear) and a batch norm.
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), bnArgs = dict(), skipConn = False, lambdaSkip = identity):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.module = nn.Linear(inSize, outSize)
        self.moduleBN = nn.BatchNorm1d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm1d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "DenseBN"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)
    

"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d).
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.

"""

class ProgConv2DBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict(), skipConn = False, lambdaSkip = identity):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        if self.skipConn:
            self.skipVar = x
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Conv2D"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["kernel_size"] = self.kernSize
        data["skip"] = self.skipConn
        return data


"""
A ProgBlock containing LSTM network, default activatin function is nn.Tanh, but can be customized easily.
The lateral layers are Dense 
"""

class ProgLstmBlock1(ProgBlock):
    def __init__(self, inSize, hidden_size, numLaterals, lat_connect, activation = nn.Tanh(), layerArgs = dict(), skipConn = False, lambdaSkip = identity):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.module = LSTM1(inSize, hidden_size)
        #self.laterals = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(numLaterals)])
        self.laterals = nn.ModuleList([LSTM_lateral(input_size=inSize, hidden_size=hidden_size, lat_connect=lat_connect) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x, hn, cn):
        if self.skipConn:
            self.skipVar = x
        return self.module(x, hn, cn)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Dense"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)
    
    def isLateralized(self):
        return True
    
class ProgLstmBlock2(ProgBlock):
    def __init__(self, inSize, hidden_size, numLaterals, lat_connect, activation = nn.Tanh(), layerArgs = dict(), skipConn = False, lambdaSkip = identity, args=None):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.skipConn = skipConn
        self.skipVar = None
        self.skipFunction = lambdaSkip
        self.module = LSTM2(inSize, hidden_size, args=args)
        #self.laterals = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(numLaterals)])
        self.laterals = nn.ModuleList([LSTM_lateral(input_size=inSize, hidden_size=hidden_size, lat_connect=lat_connect) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x, hn, cn):
        if self.skipConn:
            self.skipVar = x
        return self.module(x, hn, cn)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        if self.skipConn and self.skipVar is not None:
            x = x + self.skipFunction(self.skipVar)
        return self.activation(x)

    def getData(self):
        data = dict()
        data["type"] = "Dense"
        data["input_size"] = self.inSize
        data["output_size"] = self.outSize
        data["skip"] = self.skipConn
        return data

    def getShape(self):
        return (self.inSize, self.outSize)
    
    def isLateralized(self):
        return True


# def main():
#     # b = ProgLstmBlock(6, 64, 0).cuda()
#     # x = torch.randn((2, 6), device="cuda"). unsqueeze(1)
#     # h0 = torch.zeros(1, x.size(0), 64).cuda()
#     # c0 = torch.zeros(1, x.size(0), 64).cuda()
#     # print("x shape ", x.shape)
#     # print("h0 shape ", h0.shape)
#     # out, _, _ = b.runBlock(x, h0, c0)
#     # print("out shape ", out.shape)
#     # print(b.runActivation(out))

#     b = ProgDenseBlock(64, 4, 0).cuda()
#     x = torch.randn((2, 64), device="cuda"). unsqueeze(1)
#     h0 = torch.zeros(1, x.size(0), 64).cuda()
#     c0 = torch.zeros(1, x.size(0), 64).cuda()
#     print("x shape ", x.shape)
#     print("h0 shape ", h0.shape)
#     out = b.runBlock(x)
#     print("out shape ", out.shape)
#     print(b.runActivation(out))


# if __name__=="__main__":
#     main()