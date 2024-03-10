import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import make_padding_mask

# Encoder
class Encoder(nn.Module):
    def __init__(self,
                 num_conv,
                 conv_channels,
                 conv_kernel_size,
                 hidden_dim,
                 dropout):
        super(Encoder, self).__init__()

        # 3層1次元の畳み込み
        convolutions = nn.ModuleList()
        for _ in range(num_conv):
            convolutions.append(
                nn.Sequential(
                    nn.Conv1d(conv_channels,
                              conv_channels,
                              conv_kernel_size,
                              padding=int((conv_kernel_size - 1) // 2),
                              bias=False),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout))
            )
        self.convolutions = nn.Sequential(*convolutions)

        # 双方向LSTM
        self.bi_lstm = nn.LSTM(
            conv_channels, int(hidden_dim // 2), 1,
            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):

        # 3層1次元の畳み込みの計算
        x = self.convolutions(x.transpose(1, 2)).transpose(1, 2)

        # 双方向LSTMの計算
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        outputs, _ = self.bi_lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs

# Location Sensitive Attention
class Attention(nn.Module):
    def __init__(self,
                 encoder_dim,
                 decoder_rnn_dim,
                 hidden_dim,
                 conv_channels,
                 conv_kernel_size):
        super(Attention, self).__init__()

        self.query_layer = nn.Linear(decoder_rnn_dim, hidden_dim, bias=False)
        self.memory_layer = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.location_layer = nn.Linear(conv_channels, hidden_dim, bias=False)

        self.location_conv = nn.Conv1d(1, conv_channels,
                                       kernel_size=conv_kernel_size,
                                       padding=int((conv_kernel_size - 1) // 2),
                                       bias=False)

        self.w = nn.Linear(hidden_dim, 1)

        self.processed_memory = None

    def forward(self, query, memory, prev_attention_weights, mask):
        """
        PARAMS
        -------
        query: デコーダの隠れ状態
        memory: エンコーダからの出力
        prev_attention_weights: 前回までの累積のアテンション重み
        mask: パディング部がTrue、それ以外がFalseのマスク

        RETURNS
        -------
        attention_context: コンテキストベクトル
        attention_weights: アテンション重み
        """

        if self.processed_memory is None:
            self.processed_memory = self.memory_layer(memory)

        processed_query = self.query_layer(query.unsqueeze(1))

        processed_attention_weights = self.location_conv(prev_attention_weights.unsqueeze(1)).transpose(1, 2)
        processed_attention_weights = self.location_layer(processed_attention_weights)

        energies = self.w(torch.tanh(
            processed_query + processed_attention_weights + self.processed_memory)).squeeze(-1)

        # マスクを適用
        energies = energies.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(energies, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

# Pre-Net
class Prenet(nn.Module):
    def __init__(self,
                 decoder_output_dim,
                 num_layers,
                 hidden_dim,
                 dropout):
        super(Prenet, self).__init__()
        self.dropout = dropout

        self.prenet = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = decoder_output_dim if layer == 0 else hidden_dim
            self.prenet.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim, bias=False),
                    nn.ReLU())
            )

    def forward(self, x):
        for layer in self.prenet:
            x = F.dropout(layer(x), p=self.dropout, training=True)  # 推論時も行う
        return x

# Decoder   
class Decoder(nn.Module):
    def __init__(self,
                 num_mel_channels,
                 num_frames_per_step,  # 1ステップに推論するフレームの数
                 encoder_dim,  # エンコーダの隠れ層の次元数
                 rnn_hidden_dim,  # LSTMの次元数
                 attention_hidden_dim,  # アテンション全結合層の次元数
                 attention_conv_channels,  # アテンション畳み込み層のチャンネル数
                 attention_conv_kernel_size,  # アテンション畳み込み層のカーネルサイズ
                 prenet_num_layers,  # Pre-Netの層数
                 prenet_hidden_dim,  # Pre-Netの次元数
                 prenet_dropout,
                 max_decoder_steps,  # デコーダの出力上限、とりあえず1000に設定
                 gate_threshold,  # 終了判定
                 dropout):  # ドロップアウト率
        super(Decoder, self).__init__()
        self.num_mel_channels = num_mel_channels
        self.num_frames_per_step = num_frames_per_step
        self.encoder_dim = encoder_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.dropout = dropout

        # 注意機構
        self.attention_layer = Attention(
            encoder_dim,
            rnn_hidden_dim,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size
            )

        # Pre-Net
        self.prenet = Prenet(
            num_mel_channels * num_frames_per_step,
            prenet_num_layers,
            prenet_hidden_dim,
            prenet_dropout
            )

        # 片方向LSTM
        self.lstm1 = nn.LSTMCell(
            prenet_hidden_dim + encoder_dim, rnn_hidden_dim)

        self.lstm2 = nn.LSTMCell(
            rnn_hidden_dim + encoder_dim, rnn_hidden_dim)

        # 全結合層
        self.linear_projection = nn.Linear(
            rnn_hidden_dim + encoder_dim,
            num_mel_channels * num_frames_per_step, bias=False
            )

        self.gate_layer = nn.Linear(
            rnn_hidden_dim + encoder_dim,
            num_frames_per_step, bias=True
            )

    def forward(self, memory,
                memory_lengths,
                decoder_targets=None):
        """
        PARAMS
        -------
        memory: エンコーダの出力、(バッチサイズ, パディングされた文字列の長さ, 各文字に対するベクトル表現の次元数)
        memory_lengths: アテンションをマスクするためのエンコーダの出力長
        decoder_targets: 正解のメルスペクトログラム（Noneの場合は推論）

        RETURNS
        -------
        mel_outputs: 出力されるメルスペクトログラム(B, 80, 出力長)
        gate_outputs: 終了判定信号(B, 出力長)
        alignments: 推定場所を示すテンソル(B, 出力長, 文字数)

        MEMO
        -------
        学習時と推論時で異なる処理を行う。decoder_targetsの有無でフラグを立てて判断。
        学習時は正解のメルスペクトログラムを、推論時では一つ前の出力をprenetで処理。
        """
        inference_flag = decoder_targets is None

        B = memory.size(0)  # バッチサイズ
        max_memory_lengths = memory.size(1)  # パディング済みの文字列の長さ
        dtype = memory.dtype
        device = memory.device

        if inference_flag:
            max_decoder_steps = self.max_decoder_steps  # 推論時は出力上限1000に設定

        else:
            # (B, 80, 系列長)-> (系列長, B, 80)
            decoder_targets = decoder_targets.transpose(1, 2)

            # フレーム数の調整
            if self.num_frames_per_step > 1:
                decoder_targets = decoder_targets.view(
                    B, int(decoder_targets.size(1)/self.num_frames_per_step), -1)

            decoder_targets = decoder_targets.transpose(0, 1)
            max_decoder_steps = decoder_targets.size(0)  # 学習時の出力の長さは正解メルスペクトログラムと同じ

        # 初期化
        attention_weights = torch.zeros(B, max_memory_lengths, dtype=dtype, device=device)
        prev_attention_weights = attention_weights
        attention_context = torch.zeros(B, self.encoder_dim, dtype=dtype, device=device)

        rnn1_hidden = torch.zeros(B, self.rnn_hidden_dim, dtype=dtype, device=device)
        rnn1_cell = torch.zeros(B, self.rnn_hidden_dim, dtype=dtype, device=device)
        rnn2_hidden = torch.zeros(B, self.rnn_hidden_dim, dtype=dtype, device=device)
        rnn2_cell = torch.zeros(B, self.rnn_hidden_dim, dtype=dtype, device=device)

        # 最初の入力
        prenet_input = torch.zeros(
            B, self.num_mel_channels * self.num_frames_per_step, dtype=dtype, device=device)

        # マスクの生成
        mask = make_padding_mask(memory_lengths)

        self.attention_layer.processed_memory = None

        mel_outputs, gate_outputs, alignments = [], [], []
        t = 0

        while True:
            # Pre-Netの処理
            decoder_input = self.prenet(prenet_input)

            # 1層目LSTM
            rnn1_input = torch.cat((decoder_input, attention_context), -1)
            rnn1_hidden, rnn1_cell = self.lstm1(rnn1_input, (rnn1_hidden, rnn1_cell))
            rnn1_hidden = F.dropout(rnn1_hidden, self.dropout, self.training)

            # アテンションの計算
            attention_context, attention_weights = self.attention_layer(
                rnn1_hidden, memory, prev_attention_weights, mask)
            prev_attention_weights += attention_weights  # 累積アテンション重み

            # 2層目LSTM
            rnn2_input = torch.cat((rnn1_hidden, attention_context), -1)
            rnn2_hidden, rnn2_cell = self.lstm2(rnn2_input, (rnn2_hidden, rnn2_cell))
            rnn2_hidden = F.dropout(rnn2_hidden, self.dropout, self.training)

            # LSTMからの出力(B, 1024)とattentionからの出力(B, 512)をくっつけて全結合層に突っ込む
            decoder_hidden_attention_context = torch.cat((rnn2_hidden, attention_context), dim=1)
            mel_output = self.linear_projection(decoder_hidden_attention_context)  # (B, 80)

            gate_output = self.gate_layer(decoder_hidden_attention_context)  # 終了フラグ(B, 1)

            mel_outputs += [mel_output]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

            if inference_flag:
                prenet_input = mel_output
            else:
                prenet_input = decoder_targets[t]

            t += 1

            if t >= max_decoder_steps:
                if inference_flag:
                    print("Reached max decoder steps")
                break

            if inference_flag and (torch.sigmoid(gate_output) >= self.gate_threshold).any():
                break

        mel_outputs = torch.stack(mel_outputs, dim=2)  # (B, out_dim, 出力長)
        gate_outputs = torch.stack(gate_outputs, dim=1)  # (B, 出力長)
        alignments = torch.stack(alignments, dim=1)  # (B, 出力長, 文字数)

        if self.num_frames_per_step > 1:
            mel_outputs = mel_outputs.view(B, self.num_mel_channels, -1)  # (B, 80, 出力長)

        return mel_outputs, gate_outputs, alignments
    
# Post-Net
class Postnet(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mel_channels,
                 conv_channels,
                 conv_kernel_size,
                 dropout):
        super(Postnet, self).__init__()

        postnet = nn.ModuleList()
        for layer in range(num_layers):
            # 入力80、中間512、出力80
            in_channels = num_mel_channels if layer == 0 else conv_channels
            out_channels = num_mel_channels if layer == num_layers - 1 else conv_channels
            postnet.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels,
                              kernel_size=conv_kernel_size, stride=1,
                              padding=int((conv_kernel_size - 1) // 2),
                              dilation=1),
                    nn.BatchNorm1d(out_channels))
            )
            # 最終層はTanh関数なし
            if layer != num_layers - 1:
                postnet.append(nn.Tanh())
            postnet.append(nn.Dropout(dropout))
        self.postnet = nn.Sequential(*postnet)

    def forward(self, x):
        return self.postnet(x)


class Tacotron2(nn.Module):
    '''
    エンコーダ、デコーダ、Post-Netの処理をまとめたもの
    学習時と推論時で処理が異なる
    '''
    def __init__(self,config):
        super(Tacotron2, self).__init__()
        self.embedding = nn.Embedding(
            config.model.Tacotron2.num_symbols,
            config.model.Tacotron2.symbols_embedding_dim,
            padding_idx=0
            )

        self.encoder = Encoder(
            num_conv=config.model.Encoder.num_conv,
            conv_channels=config.model.Tacotron2.symbols_embedding_dim,
            conv_kernel_size=config.model.Encoder.conv_kernel_size,
            hidden_dim=config.model.Encoder.hidden_dim,
            dropout=config.model.Encoder.dropout
        )

        self.decoder = Decoder(
            num_mel_channels=config.model.Decoder.num_mel_channels,
            num_frames_per_step=config.model.Decoder.num_frames_per_step,
            encoder_dim=config.model.Encoder.hidden_dim,
            rnn_hidden_dim=config.model.Decoder.rnn_hidden_dim,
            attention_hidden_dim=config.model.Attention.hidden_dim,
            attention_conv_channels=config.model.Attention.conv_channels,
            attention_conv_kernel_size=config.model.Attention.conv_kernel_size,
            prenet_num_layers=config.model.Prenet.num_layers,
            prenet_hidden_dim=config.model.Prenet.hidden_dim,
            prenet_dropout=config.model.Prenet.dropout,
            max_decoder_steps=config.model.Decoder.max_decoder_steps,
            gate_threshold=config.model.Decoder.gate_threshold,
            dropout=config.model.Decoder.dropout
        )

        self.postnet = Postnet(
            num_layers=config.model.Postnet.num_layers,
            num_mel_channels=config.model.Decoder.num_mel_channels,
            conv_channels=config.model.Postnet.conv_channels,
            conv_kernel_size=config.model.Postnet.conv_kernel_size,
            dropout=config.model.Postnet.dropout
        )

    def forward(self, inputs, input_lengths, decoder_targets):
        """
        学習時の処理

        PARAMS
        -------
        inputs: 入力となるテキスト情報（音素列）
        input_lengths: 入力の長さ
        decoder_targets: 正解となるメルスペクトログラム、(B, n_mel_channels, 系列長)

        RETURNS
        -------
        mel_outputs: メルスペクトログラム
        mel_outputs_postnet: メルスペクトログラム（残差接続済み）
        gate_outputs: stop token
        alignments: アテンション重み
        -------
        """
        # 文字埋め込み
        embedding_inputs = self.embedding(inputs)

        # エンコーダによるテキストの潜在表現の獲得
        encoder_outputs = self.encoder.forward(embedding_inputs, input_lengths)

        # デコーダによるメルスペクトログラム、stop tokenの予測
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, input_lengths, decoder_targets)

        # Post-Netによる残差の予測
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        # (B, 80, 出力長) -> (B, 出力長, 80)
        mel_outputs = mel_outputs.transpose(2, 1)
        mel_outputs_postnet = mel_outputs_postnet.transpose(2, 1)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs, input_lengths):
        """
        推論時の処理
        """
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.forward(inputs, input_lengths, None)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments