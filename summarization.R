library(keras)
library(purrr)
library(stringr)
library(ggplot2)
library(tidytext)
library(rprojroot)

model_dir <-
  file.path(find_root(criterion = is_rstudio_project), "models")
code_dir <-
  file.path(find_root(criterion = is_rstudio_project), "initial_exploration")
source(file.path(code_dir, "read_data.R"))

# Purpose -----------------------------------------------------------------

# Abstractive text summarization using Keras, loosely after
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
# but using words instead of characters.



# Things to add/explore -----------------------------------------------------------

# dropout
# batch normalization
# attention


# Inspect texts and abstracts --------------------------------------------------------

# restrict to single language
# language <- "english"
language <- "german"

# how long are the texts?
# this determines max_len_text, the threshold to use for padding resp. truncating
# from here on, we work with the lower-cased, stopword-cleaned and stemmed texts
text <-
  df_all %>% filter(lang == language) %>% select(text_lower) %>% pull()
text_lengths <- text %>% map_int(str_length) %>% sort()
text_lengths %>% summary()
ggplot(data.frame(tl = text_lengths), aes(x = tl)) + geom_histogram()

# how frequent are the words?
# this determines vocab_length, the minimum frequency threshold to keep a word
df_tokens <- df_all %>% unnest_tokens(token, text_lower)
token_freqs <-
  df_tokens %>% group_by(token) %>% summarise(cnt = n()) %>% arrange(desc(cnt))
token_freqs %>% ggplot(aes(cnt)) + geom_histogram(binwidth = 20) + coord_cartesian(xlim = c(0, 250))
token_freqs$cnt %>% summary()

# how long are the abstracts?
# this determines the length of summaries to produce
abstract <-
  df_all %>% filter(lang == language) %>% select(abstract_lower) %>% pull()
abstract_lengths <- abstract %>% map_int(str_length) %>% sort()
abstract_lengths %>% summary()
ggplot(data.frame(tl = abstract_lengths), aes(x = tl)) + geom_histogram()


# Pre-process texts and abstracts --------------------------------------------------------

# preprend start token and append end token to abstracts
abstract <-
  map_chr(abstract, function(string)
    paste("STARTTOKEN", string, "ENDTOKEN"))

# tokenize

num_words <- 10000
max_len_abstract <- 50
max_len_text <- 2500

# use both text and abstracts to populate the vocabulary
text_and_abstract <- c(text, abstract)
# bug: oov_token does not work, instead, these words are just removed
# this also means: we have less input for abstracts and can choose a shorter max_len here
tok <- text_tokenizer(num_words = num_words, oov_token = "UNK")
tok %>% fit_text_tokenizer(text_and_abstract)

#tok$word_counts # how often each word occurred
#tok$word_docs # how many documents each word appeared in
#tok$word_index # mapping word -> index
tok$word_index[1:10]
#tok$document_count # how many documents we trained on

# convert texts
X <- texts_to_sequences(tok, text)
# pad to max_len_text
X <-
  pad_sequences(X,
                maxlen = max_len_text,
                padding = "post",
                truncating = "post")
dim(X)
X[1:20, 1:10]

# convert abstracts
Y <- texts_to_sequences(tok, abstract)
# pad to max_len_text
Y <-
  pad_sequences(Y,
                maxlen = max_len_abstract,
                padding = "post",
                truncating = "post")
dim(Y)
Y[1:20, 1:10]



# Inputs and hyperparameters ------------------------------------------------------------

encoder_input_data <- X
decoder_input_data <- Y
decoder_input_data[1:5, 1:10]
decoder_input_data[1:5, 41:50]
decoder_target_data <- Y[,-1] %>% cbind(0)
# decoder target data need to be one-hot-encoded
decoder_target_data <- decoder_target_data %>% to_categorical()
dim(decoder_target_data)
decoder_target_data[1, 1, 1:20]

num_decoder_tokens <- dim(decoder_target_data)[3]

vocab_length <- num_words + 1 # for 0 padding
#vocab_length <- num_words + 2 # if UNK is included
embedding_dim <- 100
dim_gru_lstm <- 50

batch_size <- 1
num_epochs <- 1

# whether to use LSTM or GRU, and whether to use bidirectional RNNs
#rnn_type <- "GRU"
#rnn_type <- "GRU_BIDIRECTIONAL"
#rnn_type <- "LSTM"
rnn_type <- "LSTM_BIDIRECTIONAL"

model_name <- file.path(model_dir, paste0("seq2seq_summarise_", rnn_type, "_", language, "_", epochs))
model_exists <- FALSE

# Model for training ----------------------------------------------------------
# for training, we feed the actual targets, offset by 1, as an additional input
# F. Chollet: "Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence."

# input layer dimension is batch size * maximum length of input text
# input text is encoded as integers
encoder_inputs  <- layer_input(shape = list(max_len_text))

# embedding takes input from 2 to 3 dimensions, the last one with size embedding_dim
encoder_inputs_embedded <-
  encoder_inputs %>% layer_embedding(input_dim = vocab_length,
                                     output_dim = embedding_dim,
                                     mask_zero = TRUE)

# the encoding RNN returns states as well as outputs
encoder_rnn <- switch(
  rnn_type,
  GRU = layer_gru(units = dim_gru_lstm, return_state =
                    TRUE),
  LSTM = layer_lstm(units = dim_gru_lstm, return_state =
                      TRUE),
  GRU_BIDIRECTIONAL = bidirectional(layer = layer_gru(
    units = dim_gru_lstm, return_state =
      TRUE
  )),
  LSTM_BIDIRECTIONAL = bidirectional(layer = layer_lstm(
    units = dim_gru_lstm, return_state =
      TRUE
  ))
)

# from the encoder RNN, all we need are the states, not the outputs
# the states will serve as context for the decoder
if (rnn_type == "GRU") {
  c(., encoder_h) %<-% (encoder_inputs_embedded %>% encoder_rnn())
} else if (rnn_type == "LSTM") {
  c(., encoder_h, encoder_c) %<-% (encoder_inputs_embedded %>% encoder_rnn())
} else if (rnn_type == "GRU_BIDIRECTIONAL") {
  c(., encoder_h_forward, encoder_h_backward) %<-% (encoder_inputs_embedded %>% encoder_rnn())
} else if (rnn_type == "LSTM_BIDIRECTIONAL") {
  c(., encoder_h_forward, encoder_c_forward, encoder_h_backward, encoder_c_backward) %<-% (encoder_inputs_embedded %>% encoder_rnn())
}


# the decoder takes as input the actual target sequence, but offset by 1 (teacher forcing)
decoder_inputs <- layer_input(shape = list(NULL))
# we also embed the abstracts
decoder_inputs_embedded <-
  decoder_inputs %>% layer_embedding(input_dim = vocab_length,
                                     output_dim = embedding_dim,
                                     mask_zero = TRUE)

# We set up our decoder to return full output sequences, and to return internal states as well.
# We don't use the return states in the training model, but we will use them in inference.
decoder_rnn <- switch(rnn_type,
                      GRU = layer_gru(
                        units = dim_gru_lstm, return_state =
                          TRUE, return_sequences = TRUE
                      ),
                      LSTM = layer_lstm(
                        units = dim_gru_lstm, return_state =
                          TRUE, return_sequences = TRUE
                      ),
                      GRU_BIDIRECTIONAL = bidirectional(layer = layer_gru(
                        units = dim_gru_lstm, return_state =
                          TRUE, return_sequences = TRUE
                      )),
                      LSTM_BIDIRECTIONAL = bidirectional(layer = layer_lstm(
                        units = dim_gru_lstm, return_state =
                          TRUE, return_sequences = TRUE
                      )))
# the decoder rnn gets its initial state from the hidden state of the encoder
# here we neglect the decoder rnn's hidden state
if (rnn_type == "GRU") {
  c(decoder_output, .) %<-% (decoder_inputs_embedded %>% decoder_rnn(initial_state = encoder_h))
} else if (rnn_type == "LSTM") {
  c(decoder_output, ., .) %<-% (decoder_inputs_embedded %>% decoder_rnn(initial_state = c(encoder_h, encoder_c)))
} else if (rnn_type == "GRU_BIDIRECTIONAL") {
  c(decoder_output, ., .) %<-% (decoder_inputs_embedded %>% decoder_rnn(initial_state = list(
    encoder_h_forward, encoder_h_backward)))
} else if (rnn_type == "LSTM_BIDIRECTIONAL") {
  c(decoder_output, ., ., ., .) %<-% (decoder_inputs_embedded %>% decoder_rnn(initial_state = list(
    encoder_h_forward, encoder_c_forward, encoder_h_backward, encoder_c_backward)))
}


# we feed the decoder rnn's output to a dense layer with softmax activation
# this output will need the target one-hot-encoded
decoder_dense <- layer_dense(units = num_decoder_tokens, activation =
                               'softmax')
decoder_output <- decoder_output %>% decoder_dense()

# the training model takes as inputs the text input to the encoder and
# the offset-by-1 abstract input to the decoder, and as output the one-hot-encoded abstracts
training_model <- keras_model(inputs = c(encoder_inputs, decoder_inputs),
                     outputs = decoder_output)
training_model

# compile model
training_model %>% compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')

# train model
if (!model_exists) {
  history <- training_model %>% fit(
    list(encoder_input_data, decoder_input_data),
    decoder_target_data,
    batch_size = batch_size,
    epochs = num_epochs,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 3),
      callback_reduce_lr_on_plateau(patience = 3),
      callback_tensorboard(log_dir = "/tmp",
                          # histogram_freq = 1,
                           write_grads = TRUE,
                           write_images = TRUE,
                           embeddings_freq = 1))
  )

  training_model %>% save_model_weights_hdf5 (paste0(model_name, "_weights.hdf5"))
  plot(history)

} else {
  training_model %>% load_model_weights_hdf5(paste0(model_name, "_weights.hdf5"))
}

# Inference model ---------------------------------------------------------
# for inference, we take as input
# at first: the start token and the hidden state from the encoder,
# and then: the last word generated and the hidden state from the decoder RNN
# accordingly, at every step we produce
# the word probabilities as well as the hidden state after running the decoder RNN

start_token <- tok$word_index$starttoken
end_token <- tok$word_index$endtoken

# the encoder model for inference takes the encoder inputs and outputs the hidden states
# here we reuse (access) the trained weights from the training model
inference_model_encoder <-
  keras_model(input = encoder_inputs, output = switch(rnn_type,
                                                      GRU = encoder_h,
                                                      LSTM = list(encoder_h, encoder_c),
                                                      GRU_BIDIRECTIONAL = list(
                                                        encoder_h_forward, encoder_h_backward),
                                                      LSTM_BIDIRECTIONAL = list(
                                                        encoder_h_forward, encoder_c_forward, encoder_h_backward, encoder_c_backward)))

inference_model_encoder

# the decoder takes as input the hidden states from the encoder
if (rnn_type == "GRU") {
  decoder_state_input_h <- layer_input(shape = c(dim_gru_lstm))
} else if (rnn_type == "LSTM") {
  decoder_state_input_h <- layer_input(shape = c(dim_gru_lstm))
  decoder_state_input_c <- layer_input(shape = c(dim_gru_lstm))
} else if (rnn_type == "GRU_BIDIRECTIONAL") {
  decoder_state_input_h_forward <- layer_input(shape = c(dim_gru_lstm))
  decoder_state_input_h_backward <- layer_input(shape = c(dim_gru_lstm))
} else if (rnn_type == "LSTM_BIDIRECTIONAL") {
  decoder_state_input_h_forward <- layer_input(shape = c(dim_gru_lstm))
  decoder_state_input_c_forward <- layer_input(shape = c(dim_gru_lstm))
  decoder_state_input_h_backward <- layer_input(shape = c(dim_gru_lstm))
  decoder_state_input_c_backward <- layer_input(shape = c(dim_gru_lstm))
}

# here we reuse (access) the decoder rnn's weights from the training model
if (rnn_type == "GRU") {
  c(decoder_output, state_h) %<-% decoder_rnn(decoder_inputs_embedded,
                                              initial_state = decoder_state_input_h)
} else if (rnn_type == "LSTM") {
  c(decoder_output, state_h, state_c) %<-% decoder_rnn(decoder_inputs_embedded,
                                                       initial_state = c(decoder_state_input_h,
                                                                            decoder_state_input_c))
} else if (rnn_type == "GRU_BIDIRECTIONAL") {
  c(decoder_output, state_h_forward, state_h_backward) %<-% decoder_rnn(decoder_inputs_embedded,
                                              initial_state = list(decoder_state_input_h_forward,
                                                                   decoder_state_input_h_backward))
} else if (rnn_type == "LSTM_BIDIRECTIONAL") {
  c(decoder_output, state_h_forward, state_c_forward, state_h_backward, state_c_backward) %<-% decoder_rnn(decoder_inputs_embedded,
                                                                       initial_state = list(decoder_state_input_h_forward,
                                                                                            decoder_state_input_c_forward,
                                                                                            decoder_state_input_h_backward,
                                                                                            decoder_state_input_c_backward))
}

# dense layer for outputting word probabilities
# here we reuse (access) the trained weights from the training model
decoder_output <-
  decoder_output %>% decoder_dense()

# the decoder model for inference takes as inputs the current sequence generated and
# the hidden state from the encoder,
# and outputs the word probabilities as well as the hidden state after running the decoder RNN
inference_model_decoder <-
  keras_model(
    inputs = switch(rnn_type,
                    GRU = c(decoder_inputs, decoder_state_input_h),
                    LSTM = c(decoder_inputs, list(decoder_state_input_h, decoder_state_input_c)),
                    GRU_BIDIRECTIONAL = c(decoder_inputs, decoder_state_input_h_forward,
                                          decoder_state_input_h_backward),
                    LSTM_BIDIRECTIONAL = c(decoder_inputs, decoder_state_input_h_forward,
                                           decoder_state_input_c_forward,
                                           decoder_state_input_h_backward,
                                           decoder_state_input_c_backward)),
    outputs = switch(rnn_type,
                     GRU = c(decoder_output, state_h),
                     LSTM = c(decoder_output, list(state_h, state_c)),
                     GRU_BIDIRECTIONAL = c(decoder_output, state_h_forward, state_h_backward),
                     LSTM_BIDIRECTIONAL = c(decoder_output, state_h_forward, state_c_forward,
                                            state_h_backward, state_c_backward))
  )

inference_model_decoder

# Inference loop ----------------------------------------------------------

#1) Encode the input sequence into state vectors.
#2) Start with a target sequence of size 1 (just the start-of-sequence character).
#3) Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character.
#4) Sample the next character using these predictions (we simply use argmax).
#5) Append the sampled character to the target sequence
#6) Repeat until we generate the end-of-sequence character or we hit the character limit.


decode_sequence <- function(input_seq) {

  # use the encoder model to get the hidden state for this input sequence
  states_value <- inference_model_encoder %>% predict(input_seq)
  # last word of generated sequence, initially that's the start token
  generated <- array(start_token, dim = c(1, 1))

  # Sampling loop - presupposes batch size of 1
  stop_condition <- FALSE
  decoded_sentence <- ""
  iteration <- 1

  # sample until either stop token is produced or the maximum summary length has been reached
  while (!stop_condition && iteration < max_len_abstract) {

    # decoder model predicts from the last hidden state (at first, the encoder's, then, its own)
    # together with the last generated token
    if (rnn_type == "GRU") {
      c(output_tokens, h) %<-% (inference_model_decoder %>% predict(list(generated, states_value)))
    } else if (rnn_type == "LSTM") {
      c(output_tokens, h, c) %<-% (inference_model_decoder %>% predict(c(generated, states_value)))
    } else if (rnn_type == "GRU_BIDIRECTIONAL") {
      c(output_tokens, h_forward, h_backward) %<-% (inference_model_decoder %>% predict(c(generated, states_value)))
    } else if (rnn_type == "LSTM_BIDIRECTIONAL") {
      c(output_tokens, h_forward, c_forward, h_backward, c_backward) %<-% (inference_model_decoder %>% predict(c(generated, states_value)))
    }

    # Sample a token
    sampled_token_index <- which.max(output_tokens[1, 1,])
    sampled_word <-
      attr(tok$word_index, "name")[[sampled_token_index]]
    decoded_sentence <- paste(decoded_sentence, sampled_word)

    # Exit condition: either hit max length or find stop character.
    if (sampled_token_index == end_token) {
      stop_condition <- True
    }

    # Update the last word of generated sequence (of length 1).
    generated <- array(sampled_token_index, dim = c(1,1))

    # update state
    if (rnn_type == "GRU") {
      states_value <- h
    } else if (rnn_type == "LSTM") {
      states_value <- list(h, c)
    } else if (rnn_type == "GRU_BIDIRECTIONAL") {
      states_value <- list(h_forward, h_backward)
    } else if (rnn_type == "LSTM_BIDIRECTIONAL") {
      states_value <- list(h_forward, c_forward, h_backward, c_backward)
    }

    iteration <- iteration + 1

  }
  decoded_sentence
}


# Try model ---------------------------------------------------------------

for (i in 1:10) {
  ## Take one sequence (part of the training test)
  ## for trying out decoding.
  input_seq <- encoder_input_data[i, , drop = FALSE]
  decoded_sentence <- decode_sequence(input_seq)
  original_text <- text[[i]]
  original_abstract <- abstract[[i]] %>% str_remove_all("STARTTOKEN|ENDTOKEN")
  cat('\nInput text:\n\n', original_text,'\n\n')
  cat('\nOriginal abstract:\n\n', original_abstract,'\n\n')
  cat('\nGenerated abstract:\n\n', decoded_sentence,'\n\n')
}
