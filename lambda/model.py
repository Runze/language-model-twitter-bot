import pickle
import numpy as np
from keras import Input
from keras.models import Model
from keras import backend as K

# from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
# detokenizer = Detok()


class LM():
    def __init__(self, model, meta_data):
        self.model = model

        # Extract meta data
        self.stoi = meta_data['stoi']
        self.itos = meta_data['itos']
        self.vocab_size = len(self.itos)
        self.UNK = meta_data['UNK']
        self.PAD = meta_data['PAD']
        self.BOS = meta_data['BOS']
        self.EOS = meta_data['EOS']
        self.max_len = meta_data['max_len']
        self.embed_size = meta_data['embed_size']
        self.hidden_size = meta_data['hidden_size']
        self.n_rnn_layers = meta_data['n_rnn_layers']
        self.dropout = meta_data['dropout']

        # Reconstruct the model with states in the output
        self.model_w_states = self.reconstruct_model()

    def create_model(self,
                     embed_layer=None,
                     rnn_layers=None,
                     dense_layer=None,
                     incl_states_in_output=False):
        # Define input tensor for X
        X = Input(shape=(None, ), name='X')
        
        # Embed
        if not embed_layer:
            embed_layer = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, mask_zero=True, name='embed_layer')
        y = embed_layer(X)
        
        # Feed into RNNs
        # Placeholders for states
        h0s, c0s = [], []
        hs, cs = [], []
        for i in range(self.n_rnn_layers):
            # Define input tensor for initial states
            h0 = Input(shape=(self.hidden_size, ), name=f'h0_{i}')
            c0 = Input(shape=(self.hidden_size, ), name=f'c0_{i}')
            
            if not rnn_layers:
                rnn_layer = layers.LSTM(self.hidden_size, return_sequences=True, return_state=True, dropout=self.dropout, recurrent_dropout=self.dropout, name=f'rnn_layer_{i}')
            else:
                rnn_layer = rnn_layers[i]
            
            y, h, c = rnn_layer(y, initial_state=[h0, c0])

            # Save states
            h0s.append(h0)
            c0s.append(c0)
            hs.append(h)
            cs.append(c)
        
        # Feed the output to the final dense layer
        if not dense_layer:
            dense_layer = layers.Dense(self.vocab_size, activation='softmax', name='dense_layer')
        y = dense_layer(y)
        
        # Put together
        inputs = [X] + h0s + c0s
        if incl_states_in_output:
            outputs = [y] + hs + cs
        else:
            outputs = y
        
        model = Model(inputs, outputs)
        print(model.summary())
        
        return model
    
    # Reconstruct the model with states in the output
    def reconstruct_model(self):
        # Extract trained layers
        embed_layer = self.model.get_layer('embed_layer')
        rnn_layers = [self.model.get_layer(f'rnn_layer_{i}') for i in range(self.n_rnn_layers)]
        dense_layer = self.model.get_layer('dense_layer')

        # Reconstruct by adding the states in the output
        model_w_states = self.create_model(embed_layer=embed_layer, rnn_layers=rnn_layers, dense_layer=dense_layer, incl_states_in_output=True)
        return model_w_states

    # Initialize states with zeros
    def init_states(self, n_obs):
        h0s, c0s = [], []
        for i in range(self.n_rnn_layers):
            zeros = np.zeros(shape=(n_obs, self.hidden_size))
            
            h0s.append(zeros)
            c0s.append(zeros)
        
        return h0s, c0s

    # Predict with a seed phrase
    def sample_w_pred_prob(self, pred_probs, temperature=.5, avoid_unk=False):
        pred_probs = pred_probs.flatten().astype('float64')
        
        # Adjust probabilities with temperature
        # https://github.com/minimaxir/textgenrnn/blob/master/textgenrnn/utils.py#L16
        if temperature == 0:
            return pred_probs.argmax()
        
        log_pred_probs = np.log(pred_probs + K.epsilon()) / temperature
        pred_probs = np.exp(log_pred_probs)
        
        if avoid_unk:
            pred_probs[self.stoi[self.UNK]] = 0
        
        pred_probs = pred_probs / pred_probs.sum()
        sample_ix = np.random.multinomial(1, pred_probs).argmax()
        return sample_ix

    def gen_seq_w_seed(self, seed_phrase, seed_h0s=None, seed_c0s=None, add_bos=False, max_chars=280):
        # Tokenize and map to indices
        seed_toks = seed_phrase.split()

        if add_bos:
            seed_toks = [self.BOS] + seed_toks
        
        seed_ixs = [self.stoi[tok] if tok in self.stoi else self.stoi[UNK] for tok in seed_toks]
        seed_ixs = np.array(seed_ixs).reshape(1, -1)

        if not seed_h0s or not seed_c0s:
            # Initialize states
            seed_h0s, seed_c0s = self.init_states(1)

        # Apply model to the seed phrase
        seed_preds = self.model_w_states.predict([seed_ixs]+seed_h0s+seed_c0s)
        
        # Extract predictions
        seed_ixs_preds, seed_hs, seed_cs = seed_preds[0], seed_preds[1:3], seed_preds[3:]

        # Extract the last predicted token and use it as the seed going forward
        seed_ix_pred = seed_ixs_preds[:, -1, :]
        seed_ix_pred = self.sample_w_pred_prob(seed_ix_pred, avoid_unk=True)
        seed_tok_pred = self.itos[seed_ix_pred]

        # Generate new words
        gen_toks = seed_toks[1:]
        chars = len(seed_phrase)
        while seed_tok_pred not in [self.PAD, self.EOS] and chars <= max_chars:
            gen_toks.append(seed_tok_pred)
            seed_h0s = seed_hs
            seed_c0s = seed_cs

            # Apply model
            seed_preds = self.model_w_states.predict([np.array(seed_ix_pred).reshape(1, -1)]+seed_h0s+seed_c0s)
            
            # Extract predictions
            seed_ix_pred, seed_hs, seed_cs = seed_preds[0], seed_preds[1:3], seed_preds[3:]
            seed_ix_pred = self.sample_w_pred_prob(seed_ix_pred, avoid_unk=True)
            seed_tok_pred = self.itos[seed_ix_pred]
            chars += len(seed_tok_pred) + 1
        
        # Detokenize
        # gen_phrase = detokenizer.detokenize(gen_toks)
        gen_phrase = ' '.join(gen_toks)

        # Output the future seed phrase and initial states too
        next_seed_phrase = gen_toks[-1] if seed_tok_pred not in [self.PAD, self.EOS] else self.BOS
        next_seed_h0s = seed_h0s
        next_seed_c0s = seed_c0s
        
        return gen_phrase, next_seed_phrase, next_seed_h0s, next_seed_c0s
