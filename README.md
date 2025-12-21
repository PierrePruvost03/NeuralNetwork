# Neural Network

## Building

`make`

this command will generate two binaries: **my_torch_generator** and **my_torch_analyzer**

## Generator

*command*: `./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]`

where nb is the number of networks you want to generate and config_file

here is an exemple of possible config file:

```
hidden_layers = [128, 64]
weight_min = -0.3
weight_max = 0.3
bias_min = -0.1
bias_max = 0.1
```

the neural network will always have 833 inputs (for every possibility on a chessboard)
and 5 outputs (Nothing, Check White, Check Black, Checkmate White, Checkmate Black)

## Analizer

...

## Contributors
| Pierre Pruvost | Kerwan Calvier | Abel Daverio |
|-----------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|
| <img src="https://github.com/PierrePruvost03.png" width="250em"/> | <img src="https://github.com/Kerwanc.png" width="250em"/> | <img src="https://github.com/abeldaverio.png" width="250em"/> |
