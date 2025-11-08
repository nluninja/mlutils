# mlutils

Utility functions to load and work with datasets and models for machine learning tasks, particularly focused on NLP and sequence labeling.

## Installation

Install the library from PyPI:

```bash
pip install dvtm-utils
```

## Usage

Import the modules into your project and call the needed utilities:

```python
from dvtm_utils import kerasutils, modelutils, dataio

# Example: Calculate model memory usage
kerasutils.get_model_memory_usage(batch_size, model) 

```



## Requirements

- Python >= 3.6
- numpy >= 1.19.0
- scikit-learn >= 0.24.0
- pandas >= 1.2.0
- tensorflow >= 2.4.0

## API Reference

### Model Utils (`modelutils`)

Utilities to work with models.

#### `compute_prediction_latency(dataset, model, n_instances=-1)`
Compute prediction latency of a model. The model must have a predict method.

**Parameters:**
- `dataset`: Input dataset for prediction
- `model`: Model with a predict method
- `n_instances`: Number of instances to test (default: -1 for all)

**Returns:** Average latency per instance

#### `from_encode_to_literal_labels(y_true, y_pred, idx2tag)`
Transform sequences of encoded labels into sequences of string labels.

**Parameters:**
- `y_true`: True labels (encoded)
- `y_pred`: Predicted labels (encoded)
- `idx2tag`: Dictionary mapping indices to tag strings

**Returns:** Tuple of (literal_y_true, literal_y_pred)

### Keras Utils (`kerasutils`)

Utilities to load embeddings, create LSTMs, and calculate memory usage. Memory functions work with TensorFlow only.

#### `get_model_memory_usage(batch_size, model)`
Return memory usage of a Keras model in MB given the batch size.

#### `print_model_memory_usage(batch_size, model)`
Print memory usage of a Keras model in MB given the batch size.

#### `load_glove_embedding_matrix(path, word_index, embed_dim)`
Load GloVe embeddings from file.

**Parameters:**
- `path`: Path to GloVe embeddings file
- `word_index`: Dictionary mapping words to indices
- `embed_dim`: Embedding dimension

**Returns:** Embedding matrix as numpy array

#### `load_w2v_nlpl_embedding_matrix(path, word_index, embed_dim)`
Load NLPL Italian word embeddings.

#### `create_BiLSTM(vocabulary_size, seq_len, n_classes, hidden_cells=128, embed_dim=32, drop=0.5, use_glove=False, glove_matrix=None)`
Create a Bidirectional LSTM model using Keras.

**Parameters:**
- `vocabulary_size`: Size of vocabulary
- `seq_len`: Sequence length
- `n_classes`: Number of output classes
- `hidden_cells`: Number of LSTM cells (default: 128)
- `embed_dim`: Embedding dimension (default: 32)
- `drop`: Dropout rate (default: 0.5)
- `use_glove`: Whether to use pre-trained embeddings (default: False)
- `glove_matrix`: Pre-trained embedding matrix (default: None)

**Returns:** Compiled Keras model

#### `create_paper_BiLSTM(vocabulary_size, seq_len, n_classes, hidden_cells=200, embed_dim=100, drop=0.4, use_glove=False, glove_matrix=None)`
Create a Bidirectional LSTM model with parameters from a specific paper.

#### `remove_flat_padding(X, y_true, y_pred, pad=0)`
Remove padding predictions and flatten the list of sequences.

#### `remove_seq_padding(X, y_true, y_pred, pad=0)`
Remove padding predictions from list of sequences while preserving sequence structure.

### I/O Utils (`dataio`)

Utilities to load datasets such as CoNLL-2003 and WikiNER.

#### `load_conll_data(filename, url_root=CONLL_URL_ROOT, dir_path='', only_tokens=False)`
Load CoNLL-2003 dataset from file or URL.

**Parameters:**
- `filename`: Name of the file to load
- `url_root`: Root URL for downloading (default: CoNLL-2003 GitHub repo)
- `dir_path`: Local directory path (default: '')
- `only_tokens`: If True, return only tokens; if False, include POS tags (default: False)

**Returns:** Tuple of (X, Y, output_labels) where X is sentences, Y is labels, and output_labels is the set of unique labels

#### `load_wikiner(path, token_only=False)`
Load WikiNER dataset from file.

**Parameters:**
- `path`: Path to WikiNER text file
- `token_only`: If True, return only tokens; if False, include POS tags (default: False)

**Returns:** Tuple of (sentences, tags, output_labels)

#### `itwac_preprocess_data(sentences)`
Preprocess text to match with the itWac embedding vocabulary (Italian).

**Parameters:**
- `sentences`: List of sentences (each sentence is a list of tokens)

**Returns:** Preprocessed sentences

#### Other functions:
- `open_read_from_url(url)`: Read a text file from URL
- `read_raw_conll(url_root, dir_path, filename)`: Read CoNLL format file
- `is_real_sentence(only_token, sentence)`: Check if a sentence is real or a document separator
- `load_anerd_data(path, filter_level='')`: Load ANERD dataset with configurable feature extraction

## Examples

### Loading CoNLL-2003 Dataset

```python
from dvtm_utils import dataio

# Load training data
X_train, y_train, labels = dataio.load_conll_data('train.txt', only_tokens=True)
print(f"Loaded {len(X_train)} sentences with {len(labels)} unique labels")
```

### Creating and Using a BiLSTM Model

```python
from dvtm_utils import kerasutils

# Create a BiLSTM model
model = kerasutils.create_BiLSTM(
    vocabulary_size=10000,
    seq_len=100,
    n_classes=9,
    hidden_cells=128,
    embed_dim=100,
    drop=0.5
)

# Check memory usage
batch_size = 32
kerasutils.print_model_memory_usage(batch_size, model)
```

### Computing Prediction Latency

```python
from dvtm_utils import modelutils

# Compute average prediction latency
latency = modelutils.compute_prediction_latency(test_data, model, n_instances=100)
print(f"Average latency: {latency:.4f} seconds per instance")
```

## Development

To set up for development:

```bash
# Clone the repository
git clone https://github.com/nluninja/mlutils.git
cd mlutils

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Author

Andrea Belli - andrea.belli@gmail.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
