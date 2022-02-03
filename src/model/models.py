import torch
from torch import nn
import torchvision
import typing as tp


class ImageEncoder(nn.Module):
    """
    Vision model. Encodes the image for attention model.
    """
    def __init__(self, count_windows: tp.Optional[int] = 4):
        """
        Initialization

        Args:
            count_windows (tp.Optional[int], optional): Count windows for attention. Defaults to 4.
        """
        super().__init__()

        efficientnet = torchvision.models.efficientnet_b0(pretrained=True)  # pretrained efficientnet_b0
        

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(efficientnet.children())[:-2]
        self.cvnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=count_windows)
        self.avg_pool = nn.AvgPool2d(kernal_size=7)
        self.fine_tune()
        # TODO add horizontal and vertical adaptive pooling [29]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward

        Args:
            images (torch.Tensor): The image converted to torch.Tensor

        Returns:
            out (torch.Tensor): Encoded image
        """
        batch_size = images.size(0)
        out = self.efficientnet(images)  # (batch_size, 1280, 7, 7)
        overal = self.avg_pool(out).permute(0, 2, 3, 1) # (batch_size, 1, 1, 1280)
        out = self.adaptive_pool(out)  # (batch_size, 1280, count_windows, count_windows)
        out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, 1280)  # (batch_size, count_windows * count_windows, 1280)
        
        return torch.cat([out, overal], dim=2) 

    def fine_tune(self, fine_tune: bool = True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        Args:
            fine_tune (bool, optional): Allow? Defaults to True.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):
    """
    Attention model. Creates the parts that the nlp model should pay attention to.
    """

    def __init__(self, encoder_dim: torch.Tensor, decoder_dim: torch.Tensor, p: tp.Optional[float] = 0.2):
        """
        Initialization

        Args:
            encoder_dim (torch.Tensor): feature size of encoded images
            decoder_dim (torch.Tensor): size of decoder's RNN
        """
        super(Attention, self).__init__()
        self.wq = nn.Linear(decoder_dim, decoder_dim, bias=False)  # create Tensor Query
        self.wk = nn.Linear(encoder_dim, decoder_dim, bias=False)  # create Tensor Key
        self.wv = nn.Linear(decoder_dim, decoder_dim, bias=False)  # create Tensor Value
        self.d = torch.sqrt(decoder_dim)  # create denominator for calculate attention_out 
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights
        self.dropout = nn.Dropout(p=p)

    def forward(self, encoder_out: torch.Tensor, decoder_hidden: torch.Tensor) -> torch.Tensor: 
        """
        Forward propagation

        Args:
            encoder_out (torch.Tensor): a tensor of dimension (batch_size, num_pixels, encoder_dim)
            decoder_hidden (torch.Tensor): previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        Returns:
            attention_out (torch.Tensor): attention weighted encoding
        """
        query = self.dropout(self.wq(decoder_hidden))  # (batch_size, seq_len, decoder_dim)

        key = self.dropout(self.wk(encoder_out))  # (batch_size, num_windows, decoder_dim)
        value = self.dropout(self.wv(encoder_out))  # (batch_size, num_windows, decoder_dim)
        energy = self.softmax(query @ key.t() / self.d)  # (batch_size, seq_len, num_windows)
        attention_out = energy @ value  # (batch_size, seq_len, decoder_dim)
        
        return attention_out

class ImageDecoder(nn.Module):
    """
    NLP model. Сreates an image description based on the attention model.
    """
    
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=1280, dropout=0.5):
        """
        Initialization

        Args:
            attention_dim ([type]): size of attention network
            embed_dim ([type]): embedding size
            decoder_dim ([type]): size of decoder's RNN
            vocab_size ([type]): size of vocabulary
            encoder_dim (int, optional): feature size of encoded images. Defaults to 1280.
            dropout (float, optional): dropout. Defaults to 0.5.
        """
        super().__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
    
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)
     
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
    def forward(self, encoder_out: torch.Tensor, encoded_captions, caption_lengths):
        """
        Forward propagation

        Args:
            encoder_out (torch.Tensor): encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
            encoded_captions ([type]): encoded captions, a tensor of dimension (batch_size, max_caption_length)
            caption_lengths ([type]): caption lengths, a tensor of dimension (batch_size, 1)

        Returns:
            predictions
            encoded_captions
            decode_lengths
            alphas
            sort_ind: 
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind