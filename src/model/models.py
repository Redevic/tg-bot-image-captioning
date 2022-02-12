import typing as tp

import torch

from random import random

from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    Vision model. Encodes the image for attention model.
    """
    def __init__(
        self, 
        model: tp.Optional[nn.Module] = models.efficientnet_b0(pretrained=True),
        model_channels_shape: tp.Optional[int] = 1280,
        model_feature_map_size: tp.Optional[int] = 7,
        n_attention_windows: tp.Optional[int] = 4,
        output_channels_shape: tp.Optional[int] = 1280,
    ):
        """
        Initialization image encode

        Args:
            model (tp.Optional[nn.Module], optional): cnn image model
            model_channels_shape (tp.Optional[int], optional): count output channels from model
            model_feature_map_size (tp.Optional[int], optional): size of output feture map from model
            n_attention_windows (tp.Optional[int], optional): count windows for attention
            output_channels_shape (tp.Optional[int], optional): count channels from image decoder
        """

        super().__init__()

        self.channels = model_channels_shape
        self.feature_map = model_feature_map_size
        self.windows = n_attention_windows
        self.output_channels = output_channels_shape

        self.model = model
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=n_attention_windows)
        self.avg_pool = nn.AvgPool2d(kernal_size=model_feature_map_size)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.linear_reshape = nn.Linear(model_channels_shape, output_channels_shape)
        # TODO add horizontal and vertical adaptive pooling [29]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward

        Args:
            image (torch.Tensor): The image converted to torch.Tensor

        Returns:
            out (torch.Tensor): Encoded image
        """

        batch_size = image.size(0)
        image_feature = self.model(image)
        # image_feature shape is (b, self.channels, self.feature_map, self.feature_map)
        avg_feature = self.avg_pool(image_feature).permute(0, 2, 3, 1).squeeze(1)
        # avg_feature shape is (b, 1, self.channels)
        windows_features = self.adaptive_pool(image_feature) 
        # windows_features shape is (b, self.channels, self.windows, self.windows)
        windows_features = windows_features.permute(0, 2, 3, 1).reshape(batch_size, -1, self.channels)
        # windows_features shape is (b, self.windows * self.windows, self.channels)
        overall_features = self.activation(torch.cat([avg_feature, windows_features], dim=1))
        # overall_features shape is (b, self.windows*self.windows + 1, self.channels)
        reshaped_overall_features = self.linear_reshape(overall_features)
        # reshaped_overall_features shape is (b, self.windows*self.windows + 1, self.output_channels)
        return reshaped_overall_features


class Attention(nn.Module):
    """
    Attention model. Creates the parts that the nlp model should pay attention to.
    """
    
    def __init__(
        self, 
        dimension: int,
        p: tp.Optional[float] = 0.2,
    ):
        """
        Initialization of attention

        Args:
            dimension (int): size of features in encoder and decoder
            p (tp.Optional[float], optional): p-value for dropout
        """

        super().__init__()

        self.dim = dimension
        self.d = torch.sqrt(dimension)  # create denominator for calculate attention_out

        self.wq = nn.Linear(dimension, dimension, bias=False)  # create Wq
        self.wk = nn.Linear(dimension, dimension, bias=False)  # create Wk
        self.wv = nn.Linear(dimension, dimension, bias=False)  # create Wv
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.dropout = nn.Dropout(p=p)

    def forward(
        self, 
        encoder_out: torch.Tensor, 
        decoder_hidden: torch.Tensor,
    ) -> torch.Tensor: 
        """
        Forward propagation

        Args:
            encoder_out (torch.Tensor): a tensor of dimension (b, count_features, features_shape)
            decoder_hidden (torch.Tensor): previous decoder output, a tensor of dimension (b, seq_len, self.dim)
        Returns:
            attention_out (torch.Tensor): attention weighted encoding
        """
        
        query = self.dropout(self.wq(decoder_hidden)) 
        # query shape is (b, seq_len, self.dim)
        key = self.dropout(self.wk(encoder_out)) 
        # key shape is (b, num_windows, self.dim)
        value = self.dropout(self.wv(encoder_out))
        # value shape is (b, num_windows, self.dim)
        energy = self.softmax(query @ key.t() / self.d)
        # energy shape is (b, seq_len, num_windows)
        weighted_encoder_out = energy @ value
        # weighted_encoder_out shape is (b, seq_len, self.dim)
        return weighted_encoder_out

class ImageDecoder(nn.Module):
    """
    NLP model. Сreates an image description based on the attention model.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int, 
        vocab_size: int,
        dropout: tp.Optional[float] = 0.5,
    ):
        """
        Initialization image decoder

        Args:
            embed_dim (tp.Optional[int]): embedding size
            hidden_dim (tp.Optional[int]): hidden dimension of decoder's RNN
            vocab_size (tp.Optional[int]): size of vocabulary
            dropout (tp.Optional[float], optional): dropout p
        """

        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.attention = Attention(hidden_dim, hidden_dim)  # attention layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim, bias=True)
        self.lm = nn.Linear(hidden_dim, vocab_size)  # linear layer to find scores over vocabulary

    def forward(
        self,
        input_token: torch.Tensor,
        hidden_state: tp.Tuple[torch.Tensor, torch.Tensor],
        encoder_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward propagation

        Args:
            input_token (tp.Optional[torch.Tensor]): input token index (b, 1)
            hidden_state (torch.Tensor): previous lstm state tuple of hidden and cell
                shape of each is (b, 1, self.hidden_dim)
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (b, num_features, self.hidden_dim)
            

        Returns:
            torch.Tensor: prediction for next token
        """
        current_hidden, _ = hidden_state
        embed_input = self.embedding(input_token)
        # embed_input shape is (b, 1, self.embed_dim)
        weighed_encoder_out = self.attention(encoder_out=encoder_out, decoder_hidden=current_hidden)
        # weighed_encoder_out shape is (b, 1, self.hidden_dim)
        concated_input = torch.cat([
            embed_input,
            weighed_encoder_out
        ], dim=2)
        # concated_input shape is (b, 1, self.hidden_dim + self.embed_dim)
        next_hidden, next_cell = self.lstm(concated_input, hidden_state)
        prediction = self.lm(next_hidden)
        # prediction shape is (b, 1, self.vocab_size)
        return prediction, (next_hidden, next_cell)



class Image2Text(nn.Module):
    def __init__(
        self,
        encoder: ImageEncoder,
        decoder: ImageDecoder,
        index_sos: tp.Optional[int] = 0,
    ):
        """
        Initialization image to text model 

        Args:
            encoder (ImageEncoder): encoder of image
            decoder (ImageDecoder): decoder of image with attention
            index_sos (int) index of <start of sentence> token
        """
        self.encoder = encoder
        self.decoder = decoder
        self.sos = index_sos
        hidden_dim = decoder.hidden_dim
        self.initializer_h = nn.Linear(hidden_dim, hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.initializer_c = nn.Linear(hidden_dim, hidden_dim)  # linear layer to find initial cell state of LSTMCell

    def forward(
        self, 
        image: torch.Tensor, 
        real_caption: torch.Tensor, 
        teacher_forcing_ratio: tp.Optional[float] = None,
    ):
        """
        Forvard text base of image

        Args:
            image (torch.Tensor): [description]
            real_caption (torch.Tensor): [description]
            teacher_forcing_ratio (tp.Optional[float], optional): [description]. Defaults to None.
        """

        batch_size = image.size(0)
        count_tokens = real_caption.size(1)
        image_features = self.encoder(image)
        mean_image_features = torch.mean(image_features, dim=1)
        hidden = self.initializer_h(mean_image_features).unsqueeze(1)
        cell = self.initializer_c(mean_image_features).unsqueeze(1)
        input_token = torch.tensor([[self.sos]]).repeat(batch_size, 1)

        predictions = torch.zeros(batch_size, count_tokens, self.decoder.vocab_size)

        for i in range(count_tokens):
            if teacher_forcing_ratio is not None and random.random() <= teacher_forcing_ratio:
                input_token = real_caption[:, i: i+1]

            prediction, (hidden, cell) = self.decoder.forward(
                input_token=input_token,
                hidden_state=(hidden, cell),
                encoder_out=image_features
            )
            predictions[:, i:i+1, :] = prediction
            input_token = prediction.argmax(-1)
        return predictions
