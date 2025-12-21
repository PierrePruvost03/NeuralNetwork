# Neural Network

## Building

`make`

This command will generate two binaries: **my_torch_generator** and **my_torch_analyzer**

## Generator

**Command:** `./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]`

Where `nb` is the number of networks you want to generate from each config file.

### Configuration File Format

**Basic example:**
```conf
# Learning parameters
learning_rate = 0.005
epochs = 250
batch_size = 64
patience = 25
train_ratio = 0.85

# Network architecture
hidden_layers = [256, 128, 64]

# Weight initialization (optional, He initialization is default)
weight_min = -0.3
weight_max = 0.3
bias_min = -0.1
bias_max = 0.1

# Dropout regularization (prevents overfitting)
dropout_rate = 0.3          # 30% dropout on all hidden layers
dropout_rates = []          # Or specify per-layer: [0.4, 0.3, 0.2, 0.0]

# Learning rate decay
lr_decay_enabled = true
lr_decay_rate = 0.9
lr_decay_step = 20
```

### Network Architecture

- **Input layer:** 833 neurons (chess position encoding)
- **Hidden layers:** Configurable via `hidden_layers`
- **Output layer:** 5 neurons
  - Nothing
  - Check White
  - Check Black
  - Checkmate White
  - Checkmate Black

### Dropout

Dropout is a regularization technique that randomly drops neurons during training to prevent overfitting:
- `dropout_rate = 0.3` means 30% of neurons are randomly dropped
- Use `0.2-0.3` for medium networks, `0.3-0.5` for large networks
- Always use `0.0` for the output layer
- Set to `0.0` if your network is underfitting

### Usage Examples

```bash
# Generate 1 network from a config
./my_torch_generator configs/optimized.conf 1

# Generate 5 networks
./my_torch_generator configs/optimized.conf 5

# Generate from multiple configs
./my_torch_generator configs/basic.conf 2 configs/large.conf 1
```

## Analyzer

**Command:** `./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE`

The analyzer can operate in two modes: **training** and **prediction**.

### Prediction Mode

Analyzes chessboards and outputs predictions.

**Command:** `./my_torch_analyzer --predict NETWORK_FILE CHESS_FILE`

**Chess file format:** Each line contains a FEN position (optionally followed by expected output for validation)
```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3
```

**Output:** One prediction per line
```
Checkmate Black
Nothing
Check Black
```

**Example:**
```bash
./my_torch_analyzer --predict my_torch_network.nn test_proper.txt
```

### Training Mode

Trains the neural network on a dataset.

**Command:** `./my_torch_analyzer --train [--save SAVEFILE] NETWORK_FILE TRAINING_FILE`

**Training file format:** Each line contains a FEN position followed by the expected output
```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
```

**Options:**
- `--save SAVEFILE`: Save the trained network to a different file (default: overwrites LOADFILE)

**Examples:**
```bash
# Train and overwrite original network
./my_torch_analyzer --train my_network.nn train_proper.txt

# Train and save to new file
./my_torch_analyzer --train --save trained_network.nn my_network.nn train_proper.txt
```

### Output Classes

The analyzer predicts one of 5 possible states:
- **Nothing** - No check or checkmate
- **Check White** - White king is in check
- **Check Black** - Black king is in check
- **Checkmate White** - White is checkmated (Black wins)
- **Checkmate Black** - Black is checkmated (White wins)

## Contributors
| Pierre Pruvost | Kerwan Calvier | Abel Daverio |
|-----------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| <img src="https://github.com/PierrePruvost03.png" width="250em"/> | <img src="https://github.com/Kerwanc.png" width="250em"/> | <img src="https://github.com/abeldaverio.png" width="250em"/> |
