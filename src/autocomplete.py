import pickle
import numpy as np

MODEL_PATH = "index/rnn_model.keras"
CHAR_MAP_PATH = "index/rnn_chars.pkl"

SEQ_LENGTH = 5
MAX_TRAINING_CHARS = 300000  
EPOCHS = 20
BATCH_SIZE = 64

_model = None
_char_to_index = None
_index_to_char = None


def build_training_data(text, seq_length):
    chars = sorted(set(text))
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for i, c in enumerate(chars)}

    X = []
    y = []
    for i in range(len(text) - seq_length):
        X.append([char_to_index[c] for c in text[i:i + seq_length]])
        y.append(char_to_index[text[i + seq_length]])

    X = np.array(X).reshape((len(X), seq_length, 1))
    y = np.array(y)
    return X, y, char_to_index, index_to_char


def build_rnn_model(seq_length, num_chars):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense

    model = Sequential()
    model.add(SimpleRNN(50, activation="relu", input_shape=(seq_length, 1)))
    model.add(Dense(num_chars, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_save_rnn(df, seq_length=SEQ_LENGTH):
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    text = " ".join(shuffled["text"].tolist())
    if len(text) > MAX_TRAINING_CHARS:
        text = text[:MAX_TRAINING_CHARS]

    X, y, char_to_index, index_to_char = build_training_data(text, seq_length)
    model = build_rnn_model(seq_length, len(char_to_index))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    model.save(MODEL_PATH)
    with open(CHAR_MAP_PATH, "wb") as f:
        pickle.dump(
            {"char_to_index": char_to_index, "index_to_char": index_to_char, "seq_length": seq_length},
            f,
        )


def load_rnn():
    global _model, _char_to_index, _index_to_char
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
        with open(CHAR_MAP_PATH, "rb") as f:
            maps = pickle.load(f)
        _char_to_index = maps["char_to_index"]
        _index_to_char = maps["index_to_char"]
    return _model, _char_to_index, _index_to_char


def sample_with_temperature(probabilities, temperature=0.8):
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities + 1e-8) / temperature
    exp_probs = np.exp(probabilities)
    probabilities = exp_probs / np.sum(exp_probs)
    return np.random.choice(len(probabilities), p=probabilities)


def suggest_completion(prompt, max_chars=15, temperature=0.8):
    model, char_to_index, index_to_char = load_rnn()
    seq_length = SEQ_LENGTH

    if not prompt:
        raise ValueError("Cannot predict a completion for an empty prompt")

    unknown = sorted(set(c for c in prompt if c not in char_to_index))
    if unknown:
        raise ValueError(f"Cannot predict: characters not seen during training: {unknown}")

    pad_char = " " if " " in char_to_index else prompt[0]
    context = prompt[-seq_length:].rjust(seq_length, pad_char)

    generated = ""
    for _ in range(max_chars):
        input_seq = np.array([[char_to_index[c]] for c in context]).reshape((1, seq_length, 1))
        predicted = model.predict(input_seq, verbose=0)[0]
        predicted_index = sample_with_temperature(predicted, temperature)
        predicted_char = index_to_char[predicted_index]

        if predicted_char == " ":
            break

        generated += predicted_char
        context = context[1:] + predicted_char

    return generated