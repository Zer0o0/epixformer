# Model architecture
#
import numpy as np
import tensorflow as tf
import tensorflow_text as text

# Embedding part
def PositionalEncoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis] 
    depths = np.arange(depth)[np.newaxis, :]/depth 
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates 
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=False)
        self.pos_encoding = PositionalEncoding(length=1024, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

# Attention part
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

# Feedforward part
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads,
                 dff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = PositionalEmbedding(
            vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        # Add dropout.
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)  # (batch_size, seq_len, d_model)
        return x  


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 *,
                 d_model,
                 num_heads,
                 dff,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        #
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.cross_attention(x=x, context=context)
        # Cache the last attention scores
        self.last_attn_scores = self.cross_attention.last_attn_scores

        x = self.ffn(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, len_motif,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.conv_layer =tf.keras.layers.Conv1D(filters=24, kernel_size=len_motif, activation='relu')
        # self.pool=tf.keras.layers.MaxPooling1D(pool_size=2)
        # self.dropout=tf.keras.layers.Dropout(rate=0.01)
        self.conn_layer=tf.keras.layers.Dense(units=d_model,activation='relu')

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,
                         dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]

        self.last_attn_scores = None

    def call(self, x, context):
        x = self.conv_layer(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        x = tf.transpose(x,perm=[0, 2, 1])
        x = self.conn_layer(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


class EIformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff,
                 vocab_size, len_motif, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               len_motif=len_motif,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1,),
        ])

    def call(self, inputs):
        # here,the inputs is (dna_info, epi_info)
        # dna_info, one-hot matrix of DNA sequence, with shape (200,4),
        # epi_info, tokens of epigenetics info, with shape (200,)
        #
        x, context = inputs 
        context = self.encoder(context)
        x = self.decoder(x, context)
        # Final linear layer output.
        logits = self.final_layer(x)

        return logits
