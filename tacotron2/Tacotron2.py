import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 全結合層＋初期化（一様分布）
class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
        
    def forward(self, x):
        return self.linear_layer(x)

# 1次元畳み込み層＋初期化（一様分布）
class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)
        
        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init_gain))
        
    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class LocationLayer(nn.Module):
    def __init__(self,
                 attention_n_filters,
                 attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

def get_mask_from_lengths(lengths):
        """
        MEMO
        -------
        torch.maxは与えられたテンソルの最大値を返す関数
        .item()でテンソルの要素をpython組み込み型として取得
        torch.arangeは第一引数から第二引数まで第三引数間隔のテンソルが生成される
        """
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
        mask = (ids < lengths.unsqueeze(1)).byte()
        mask = torch.le(mask, 0)
        return mask

# エンコーダ
class Encoder(nn.Module):
    def __init__(self,
                 num_vocab=66,
                 encoder_n_convolutions=3,  # 3層
                 encoder_embedding_dim=512,
                 encoder_kernel_size=5
                 ):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding(num_vocab, encoder_embedding_dim, padding_idx=0)

        # 3層1次元の畳み込み
        self.convolutions = nn.ModuleList()
        for _ in range(encoder_n_convolutions):
            self.convolutions += [
                ConvNorm(encoder_embedding_dim,
                         encoder_embedding_dim,
                         encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) // 2),
                         bias=False, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        self.convolutions = nn.Sequential(*self.convolutions)

        # 双方向LSTM
        self.blstm = nn.LSTM(encoder_embedding_dim,
                             int(encoder_embedding_dim // 2), 1,
                             batch_first=True, bidirectional=True)
        
    def forward(self, x, input_lengths):

        x = self.embed(x)

        # 3層1次元の畳み込みの計算
        x = self.convolutions(x.transpose(1, 2)).transpose(1, 2)

        # 双方向LSTMの計算
        x = pack_padded_sequence(x, input_lengths, batch_first=True)
        outputs, _ = self.blstm(x)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        return outputs

# Location Sensitive Attention
class Attention(nn.Module):
    def __init__(self,
                 attention_rnn_dim=1024,
                 encoder_embedding_dim=512,
                 attention_dim=128,
                 attention_conv_channels=32,
                 attention_conv_kernel_size=31):
        super(Attention, self).__init__()
        
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')  # query1つ目
        self.memory_layer = LinearNorm(encoder_embedding_dim, attention_dim, bias=False, w_init_gain='tanh')  # query2つ目
        self.location_layer = LocationLayer(attention_conv_channels,
                                            attention_conv_kernel_size,
                                            attention_dim)
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.score_mask_value = -float("inf")
        
    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        アテンションエネルギーの獲得

        PARAMS
        -------
        query: デコーダからの出力
        processed_memory: 処理済みのエンコーダからの出力、エンコーダからの出力を全結合層で処理したものだが、値変わらないのでデコーダ実行時に最初に処理
        attention_weights_cat: 

        RETURNS
        -------
        energies:
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)

        alignment = alignment.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

# Pre-Net
class Prenet(nn.Module):
    def __init__(self,
                 in_dim=80,  # n_mel_channels=80 * n_frames_per_step=1
                 layers=2,
                 hidden_dim=256):
        super(Prenet, self).__init__()
        prenet = nn.ModuleList()

        for layer in range(layers):
            in_channels = in_dim if layer == 0 else hidden_dim
            prenet += [
                LinearNorm(in_channels, hidden_dim, bias=False),
                nn.ReLU(),
            ]
        self.prenet = nn.Sequential(*prenet)
        
    def forward(self, x):
        for layer in self.prenet:
            x = F.dropout(layer(x), p=0.5, training=True)  # 推論時でも適応
        return x

# Post-Net
class Postnet(nn.Module):
    def __init__(self,
                 in_dim=80,
                 layers=5,
                 embedding_dim=512,
                 kernel_size=5,
                 dropout=0.5):
        super(Postnet, self).__init__()
        postnet = nn.ModuleList()

        for layer in range(layers):
            in_channels = in_dim if layer == 0 else embedding_dim
            out_channels = in_dim if layer == layers - 1 else embedding_dim
            w_init_gain = 'linear' if layer == layers -1 else 'tanh'
            postnet += [
                ConvNorm(in_channels, out_channels, 
                         kernel_size=kernel_size, stride=1,
                         padding=int((kernel_size - 1) / 2),
                         dilation=1, w_init_gain=w_init_gain),
                nn.BatchNorm1d(out_channels)
            ]
            if layer != layers - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(dropout)]
        self.postnet = nn.Sequential(*postnet)
          
    def forward(self, x):
        return self.postnet(x)

class Decoder(nn.Module):
    def __init__(self,
                 n_mel_channels=80,
                 n_frames_per_step=1,
                 encoder_hidden_dim=512,  # エンコーダの隠れ層の次元数
                 hidden_dim=1024,  # LSTMの次元数
                 prenet_layers=2,  # Pre-Netの層数
                 prenet_hidden_dim=256,  # Pre-Netの隠れ層の次元数
                 attention_dim=128,  # アテンション層全結合層の次元数
                 attention_rnn_dim=1024,  # アテンション全結合層の次元数
                 attention_conv_channels=32,  # アテンション畳み込みのチャンネル数
                 attention_conv_kernel_size=31,  # アテンション畳み込みのカーネルサイズ
                 inference_max_decoder_steps=1000,  # デコーダの出力上限、とりあえず1000に設定
                 gate_threshold=0.5,  # 終了判定
                 p_dropout=0.1):
        super(Decoder, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.prenet_layers = prenet_layers
        self.prenet_hidden_dim = prenet_hidden_dim
        self.attention_dim = attention_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.attention_conv_channels = attention_conv_channels
        self.attention_conv_kernel_size = attention_conv_kernel_size
        self.inference_max_decoder_steps = inference_max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_dropout = p_dropout

        # 注意機構
        self.attention_layer = Attention(attention_rnn_dim, encoder_hidden_dim,
                                         attention_dim, attention_conv_channels,
                                         attention_conv_kernel_size)
        
        # Pre-Net
        self.prenet = Prenet(n_mel_channels * n_frames_per_step,
                             prenet_layers, prenet_hidden_dim)

        # 片方向LSTM
        self.attention_rnn = nn.LSTMCell(
            prenet_hidden_dim + encoder_hidden_dim, hidden_dim)
        
        self.decoder_rnn = nn.LSTMCell(
            hidden_dim + encoder_hidden_dim, hidden_dim)

        # 出力へのプロジェクション層
        proj_in_dim = encoder_hidden_dim + hidden_dim
        self.proj_out = LinearNorm(proj_in_dim, n_mel_channels * n_frames_per_step, bias=False)
        self.gate_layer = LinearNorm(proj_in_dim, 1, bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """
        最初のデコーダ入力として使用する、すべてゼロのフレームを取得

        PARAMS
        -------
        memory: エンコーダの出力

        RETURNS
        -------
        decoder_input: すべてが0のフレーム
        """
        B = memory.size(0)
        frame_size = self.n_mel_channels * self.n_frames_per_step
        dtype = memory.dtype
        device = memory.device
        decoder_input = memory.new_zeros(B, frame_size, dtype=dtype, device=device)
        return decoder_input

    def initialize_decoder_states(self, memory):
        """
        デコーダの状態を初期化する関数

        PARAMS
        -------
        memory: エンコーダの出力
        mask: 

        MEMO
        -------
        torch.zeroはスカラー値0で満たされたテンソルを返す
        """
        B = memory.size(0)  # バッチサイズ
        MAX_TIME = memory.size(1)  # パディング済みの文字列の長さ
        dtype = memory.dtype
        device = memory.device

        attention_hidden = torch.zeros(B, self.attention_rnn_dim, dtype=dtype, device=device)
        attention_cell = torch.zeros(B, self.attention_rnn_dim, dtype=dtype, device=device)

        decoder_hidden = torch.zeros(B, self.hidden_dim, dtype=dtype, device=device)
        decoder_cell = torch.zeros(B, self.hidden_dim, dtype=dtype, device=device)

        attention_weights = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_weights_cum = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        attention_context = torch.zeros(B, self.encoder_hidden_dim, dtype=dtype, device=device)

        processed_memory = self.attention_layer.memory_layer(memory)

        return (attention_hidden, attention_cell, decoder_hidden,
                decoder_cell, attention_weights, attention_weights_cum,
                attention_context, processed_memory)


    def parse_decoder_inputs(self, decoder_inputs):
        """
        デコーダ入力の準備

        PARAMS
        -------
        decoder_inputs: 学習時におけるデコーダの入力

        RETURNS
        -------
        decoder_inputs: 処理済みの入力

        MEMO
        -------
        decoder_inputsの次元は(バッチサイズ, )
        .transposeは、指定された次元を入れ替える
        .viewは、配列の形状を変更する
        """

        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels) T_outはおそらくメルスペクトログラムの出力長（時間軸方向の長さ）
        decoder_inputs = decoder_inputs.transpose(1, 2)  # 1次元目と2次元目の入れ替え
        # おそらく、同時に予測するフレーム数を増やした場合に系列長を削減する処理、n_frames_per_step=1では意味ない
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),  # バッチサイズ
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)  # 0次元目と1次元目の入れ替え
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """
        デコーダの出力の準備

        PARAMS
        -------
        mel_outputs: 出力のメルスペクトログラム
        gate_outputs: 終了判断のフラグ
        alignments: 位置合わせ用のテンソル

        RETURNS
        -------
        処理済みパラメータ3つ
        mel_outputs  (バッチサイズ, 80, 時間軸方向のデータ数)
        gate_outputs
        alignments

        MEMO
        -------
        .contiguous()を実行すると要素の順序が連続？になる。

        """
        # (T_out, B) -> (B, T_out) T_outはおそらく時間軸方向のデータ数
        alignments = alignments.transpose(0, 1).contiguous()
        # (T_out, B) -> (B, T_out)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        # メルスペクトログラムのリサイズ
        shape = (mel_outputs.shape[0], -1, self.n_mel_channels)  # (バッチサイズ, 時間軸方向のデータ数, メル長) の形を作る
        mel_outputs = mel_outputs.view(*shape)  # 多分n_frame_per_step=1では意味ない
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self,
               decoder_input,
               attention_hidden,
               attention_cell,
               decoder_hidden,
               decoder_cell,
               attention_weights,
               attention_weights_cum,
               attention_context,
               memory,
               processed_memory,
               mask):
        """
        デコーダの1ステップの処理

        PARAMS
        -------
        decoder_input: 1ステップ前の出力

        RERURNS
        -------
        mel_output
        gate_output
        attention_weights

        MEMO
        -------
        torch.catは与えられたテンソルを結合する
        """
        # pre-net通過した前回の出力メル(1, 256)とattentionからの出力(1, 512)をくっつけてLSTMに突っ込む
        cell_input = torch.cat((decoder_input, attention_context), -1)  # -1で最後の次元に結合
        attention_hidden, attention_cell = self.attention_rnn(cell_input, (attention_hidden, attention_cell))  # 1コ前の出力も入力として加える
        attention_hidden = F.dropout(attention_hidden, self.p_dropout, self.training)

        # attentionの計算
        attention_weights_cat = torch.cat((attention_weights.unsqueeze(1), attention_weights_cum.unsqueeze(1)), dim=1)
        attention_context, attention_weights = self.attention_layer(attention_hidden, memory, processed_memory, attention_weights_cat, mask)
        attention_weights_cum += attention_weights  # 推定場所を示すテンソル
        
        # 1つ目のLSTMからの出力(1, 1024)とattentionからの出力(1, 512)をくっつけてLSTMに突っ込む
        decoder_input = torch.cat((attention_hidden, attention_context), -1)
        decoder_hidden, decoder_cell = self.decoder_rnn(decoder_input, (decoder_hidden, decoder_cell))
        decoder_hidden = F.dropout(decoder_hidden, self.p_dropout, self.training)

        # LSTMからの出力(1, 1024)とattentionからの出力(1, 512)をくっつけてプロジェクション層に突っ込む
        decoder_hidden_attention_context = torch.cat((decoder_hidden, attention_context), dim=1)
        decoder_output = self.proj_out(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)  # 終了フラグ

        return (decoder_output, gate_prediction, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights,
                attention_weights_cum, attention_context)

    def forward(self, memory,
                memory_lengths,
                decoder_targets=None):
        """
        PARAMS
        -------
        memory: エンコーダの出力、(バッチサイズ, パディングされた文字列の長さ, 各文字に対するベクトル表現の次元数)
        memory_lengths: アテンションをマスクするためのエンコーダの出力長
        decoder_targets: 正解のメルスペクトログラム

        RETURNS
        -------
        mel_outputs: 出力されるメルスペクトログラム(1, 80)
        gate_outputs: 終了判定信号(1, 1)
        alignments: 推定場所を示すテンソル(1, 文字数)

        MEMO
        -------
        推論時では一つ前の出力をprenetにつっこむが、学習時には一番最初は全てゼロのテンソルを、次回以降は正解のメルスペクトログラムをprenetにつっこむ
        """
        is_inference = decoder_targets is None

        prenet_input = self.get_go_frame(memory)

        if is_inference:
            max_decoder_steps = self.inference_max_decoder_steps  # 推論時は出力上限1000に設定
        else:
            decoder_targets = self.parse_decoder_inputs(decoder_targets) # 正解のメルスペクトログラム
            max_decoder_steps = decoder_targets.size(0)  # 学習時の出力の長さは正解メルスペクトログラムと同じ

        mask = get_mask_from_lengths(memory_lengths).to(memory.device)

        (attention_hidden,
         attention_cell,
         decoder_hidden,
         decoder_cell,
         attention_weights,
         attention_weights_cum,
         attention_context,
         processed_memory) = self.initialize_decoder_states(memory)  # 初期化

        mel_outputs, gate_outputs, alignments = [], [], []
        t = 0

        while True:
            decoder_input = self.prenet(prenet_input)

            (mel_output,
            gate_output,
            attention_hidden,
            attention_cell,
            decoder_hidden,
            decoder_cell,
            attention_weights,
            attention_weights_cum,
            attention_context) = self.decode(decoder_input,
                                             attention_hidden,
                                             attention_cell,
                                             decoder_hidden,
                                             decoder_cell,
                                             attention_weights,
                                             attention_weights_cum,
                                             attention_context,
                                             memory,
                                             processed_memory,
                                             mask)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]

            if is_inference:
                prenet_input = mel_output
            else:
                prenet_input = decoder_targets[t]

            t += 1
            if t >= max_decoder_steps:
                if is_inference:
                    print("Warning! Reached max decoder steps")
                break
            if is_inference and (torch.sigmoid(gate_output) >= self.gate_threshold).any():
                break

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            torch.stack(mel_outputs),
            torch.stack(gate_outputs),
            torch.stack(alignments))

        return mel_outputs, gate_outputs, alignments

class Tacotron2(nn.Module):
    '''
    '''
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, inputs, input_lengths, decoder_targets):
        """
        PARAMS
        -------
        inputs: 入力となるテキスト情報（音素列）
        input_lengths: 入力の長さ
        decoder_targets: 正解となるメルスペクトログラム、(B, n_mel_channels, T_out)

        RETURNS
        -------
        """
        # エンコーダによるテキストの洗剤表現の獲得
        encoder_outputs = self.encoder(inputs, input_lengths)

        # デコーダによるメルスペクトログラム、stop tokenの予測
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, input_lengths, decoder_targets)

        # Post-Netによる残差の予測
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        # (B, C, T) -> (B, T, C)
        mel_outputs = mel_outputs.transpose(2, 1)
        mel_outputs_postnet = mel_outputs_postnet.transpose(2, 1)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs, input_lengths):
        """
        """

        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.forward(inputs, input_lengths, None)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments