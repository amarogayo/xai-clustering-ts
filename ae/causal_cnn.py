import torch

class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """
    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        # print("Input SqueezeChannels", x.shape)
        # print("Output SqueezeChannels", x.squeeze(2).shape)
        return x.squeeze(2)

class UnsqueezeChannels(torch.nn.Module):
    def __init__(self):
        super(UnsqueezeChannels, self).__init__()

    def forward(self, x):
        # print("Input UnsqueezeChannels", x.shape)
        # print("Output UnsqueezeChannels", x.unsqueeze(2).shape)
        return x.unsqueeze(2) 

class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)

class CausalConvolutionBlockReverse(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation,
                 final=False):
        super(CausalConvolutionBlockReverse, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
            out_channels, in_channels, kernel_size,
            padding=0, dilation=dilation
        ))

        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.ConvTranspose1d(
            out_channels, out_channels, kernel_size,
            padding=0, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()
        
        # Residual connection
        self.upordownsample = torch.nn.ConvTranspose1d(
           out_channels, in_channels, 1
        ) if in_channels != out_channels else None

        # Causal network decoder
        self.causal = torch.nn.Sequential(
            conv2, chomp2, relu2, conv1, chomp1, relu1
            )
        
        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        # print("Decoder block output:", out_causal.size())
        # print("Tensor decoder:", out_causal)
        return out_causal
 
class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, out_channels,
                 kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size
        
        # print("depth:",depth)
        for i in range(depth):
            # print("dialation_size:",dilation_size)
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(
                in_channels_block, channels, kernel_size, dilation_size
            )]
            dilation_size *= 2  # Doubles the dilation size at each step
        
        # print("dialation_size:",dilation_size)
        # Last layer
        layers += [CausalConvolutionBlock(
            channels, out_channels, kernel_size, dilation_size
        )]
        # print("layers:", len(layers))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # print("Input CausalCNN:", x.size())
        # print("Output CausalCNN:", self.network(x).size())
        return self.network(x)


class CausalCNNDecode(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, out_channels,
            kernel_size):
        super(CausalCNNDecode, self).__init__()

        layers = []
        dialation_size = pow(2,depth) #last dialization size from encoder
       
        layers += [CausalConvolutionBlockReverse(
            channels, out_channels, kernel_size, dialation_size
        )]
        
        # print("layers:", len(layers))
        # print("in_channels:", in_channels)
        # print("depth:", depth)
        
        # reverse building of CausalConvolutionBlocks
        for i in range(depth):
            # print("dialation_size:",dialation_size)
            # print("i",i)
            dialation_size = int(dialation_size / 2)

            in_channels_block = in_channels if i  == depth-1  else channels
           
            layers += [CausalConvolutionBlockReverse(
                in_channels_block, channels, kernel_size, dialation_size
            )]

        # print("dialation_size:",dialation_size)
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # print("Input CausalCNNDecode:", x.size())
        # print("Output CausalCNNDecode:", self.network(x).size())
        return self.network(x)

class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, x):
        # print("x CausalCNNEncoder", x.size())
        return self.network(x)

class CausalCNNDecoder(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size,
            out_channles, kernel_size):
        super(CausalCNNDecoder, self).__init__()

        squeeze = SqueezeChannels()
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        causal_cnn_decode = CausalCNNDecode(
                in_channels, channels, depth, reduced_size, kernel_size
        )
       
        # Reverse order of CausalCNNEncoder
        self.network = torch.nn.Sequential(
                squeeze, reduce_size, causal_cnn_decode 
        )
    
    def forward(self, x):
        return self.network(x)

class CausalCNNAutoencoder(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size,
            out_channels, kernel_size):
        super(CausalCNNAutoencoder, self).__init__()
        
        reduce_size = torch.nn.AdaptiveMaxPool1d(1,return_indices=True)
       
        # Causal CNN Encoder
        causal_cnn = CausalCNN(
            in_channels, channels, depth, reduced_size, kernel_size
        )

        # Causal CNN Decoder
        causal_cnn_decode = CausalCNNDecode(
            in_channels, channels, depth, reduced_size, kernel_size
        )
        
        # Encoding
        self.encoder = torch.nn.Sequential(
            causal_cnn, reduce_size
        )

        self.squeeze = SqueezeChannels()
        self.linear = torch.nn.Linear(reduced_size, out_channels)
        
        # Decoding
        self.linear_2 = torch.nn.Linear(out_channels, reduced_size)
        self.unsqueeze = UnsqueezeChannels()
        self.unpool = torch.nn.MaxUnpool1d(1113)

        self.decoder = torch.nn.Sequential(
            causal_cnn_decode
        )

    def forward(self, x):
        # For an initial better understanding, use print statements
        input_size = x.size()
        
        # Encoder
        out, indices = self.encoder(x)

        # MaxPooling output and indices
        max_pooling_out = out
        max_pooling_indices = indices

        # Squeeze
        out = self.squeeze(out)

        # Linear layers
        out = self.linear(out)
        out = self.linear_2(out)

        # Unsqueeze
        out = self.unsqueeze(out)

        # MaxUnpooling
        # 1113 = length of the longest anomaly
        out = self.unpool(out, indices, output_size=(26,84,1113))

        # Decoder
        x_hat = self.decoder(out)

        # Loss
        loss = torch.nn.MSELoss(reduction='none')
        
        # Output
        output = loss(x_hat, x)
        
        # SHAP requires a single loss per sample, so we sum over the time and feature dimensions (1,2) 
        output = torch.sum(output, dim=(1,2))
        output = output.unsqueeze(1)

        # Output has a size of (batch_size,1)
        return output
