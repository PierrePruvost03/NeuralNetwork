use std::fs;
use std::vec;

use crate::network::datastruct::layer::Layer;

pub struct Network(pub Vec<Layer>);

impl Network {
    pub fn new(config: String) -> Result<Self, String> {
        Ok(Network(
            config
                .split("\n---\n")
                .map(|line| Layer::new(String::from(line)).unwrap())
                .collect(),
        ))
    }

    pub fn new_random_he(
        mut nb_input: u32,
        nb_perceptron: Vec<u32>,
        dropout_rates: Vec<f64>,
        output_activation: &str,
    ) -> Network {
        let num_layers = nb_perceptron.len();
        let mut layers: Vec<Layer> = nb_perceptron
            .iter()
            .enumerate()
            .map(|(idx, &nb)| {
                let activation = if idx == num_layers - 1 {
                    output_activation
                } else {
                    "relu" // Use ReLU for hidden layers
                };

                let layer = Layer::new_random_he(nb, nb_input, activation);
                nb_input = nb;
                layer
            })
            .collect();

        for (idx, layer) in layers.iter_mut().enumerate() {
            if idx < dropout_rates.len() {
                layer.set_dropout(dropout_rates[idx]);
            }
        }

        Network(layers)
    }

    pub fn set_training_mode(&mut self, training: bool) {
        for layer in &mut self.0 {
            layer.set_training_mode(training);
        }
    }

    pub fn exec(&self, inputs: Vec<f64>) -> Vec<f64> {
        let mut current_inputs = inputs;
        for layer in &self.0 {
            current_inputs = layer.exec(current_inputs);
        }
        current_inputs
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<Vec<f64>> {
        let mut all_outputs: Vec<Vec<f64>> = vec![inputs.to_vec()];

        for layer in &self.0 {
            let new_output = layer.forward(&all_outputs.last().unwrap());
            all_outputs.push(new_output);
        }
        return all_outputs;
    }

    #[allow(dead_code)]
    pub fn train(&mut self, inputs: &Vec<f64>, targets: &Vec<f64>, learning_rate: f64) {
        let all_outputs = self.forward(inputs);
        let mut deltas: Vec<Vec<f64>> = vec![];

        deltas.push(
            self.0
                .last()
                .unwrap()
                .backward_output(all_outputs.last().unwrap(), targets),
        );
        for index in (0..self.0.len() - 1).rev() {
            deltas.push(self.0[index].backward_hidden(
                &all_outputs[index + 1],
                deltas.last().unwrap(),
                &self.0[index + 1],
            ));
        }

        deltas.reverse();

        for index in 0..self.0.len() {
            self.0[index].update_weights(&deltas[index], &all_outputs[index], learning_rate);
        }
    }

    pub fn train_softmax_ce(&mut self, inputs: &Vec<f64>, targets: &Vec<f64>, learning_rate: f64) {
        let all_outputs = self.forward(inputs);
        let mut deltas: Vec<Vec<f64>> = vec![];

        let output_layer_outputs = all_outputs.last().unwrap();
        let softmax_outputs = Self::softmax(output_layer_outputs);
        let output_deltas: Vec<f64> = softmax_outputs
            .iter()
            .zip(targets.iter())
            .map(|(s, t)| t - s)
            .collect();

        deltas.push(output_deltas);

        for index in (0..self.0.len() - 1).rev() {
            deltas.push(self.0[index].backward_hidden(
                &all_outputs[index + 1],
                deltas.last().unwrap(),
                &self.0[index + 1],
            ));
        }

        deltas.reverse();

        for index in 0..self.0.len() {
            self.0[index].update_weights(&deltas[index], &all_outputs[index], learning_rate);
        }
    }

    pub fn train_batch_softmax_ce(
        &mut self,
        batch: &Vec<(Vec<f64>, Vec<f64>)>,
        learning_rate: f64,
    ) {
        if batch.is_empty() {
            return;
        }

        let num_layers = self.0.len();

        let mut accumulated_weight_gradients: Vec<Vec<Vec<f64>>> = vec![];
        let mut accumulated_bias_gradients: Vec<Vec<f64>> = vec![];

        for layer_idx in 0..num_layers {
            let num_neurons = self.0[layer_idx].neurons.len();
            let num_inputs = if layer_idx == 0 {
                batch[0].0.len()
            } else {
                self.0[layer_idx - 1].neurons.len()
            };

            let weight_grads: Vec<Vec<f64>> = vec![vec![0.0; num_inputs]; num_neurons];
            let bias_grads: Vec<f64> = vec![0.0; num_neurons];

            accumulated_weight_gradients.push(weight_grads);
            accumulated_bias_gradients.push(bias_grads);
        }

        for (inputs, targets) in batch {
            let all_outputs = self.forward(inputs);
            let mut deltas: Vec<Vec<f64>> = vec![];

            let output_layer_outputs = all_outputs.last().unwrap();
            let softmax_outputs = Self::softmax(output_layer_outputs);
            let output_deltas: Vec<f64> = softmax_outputs
                .iter()
                .zip(targets.iter())
                .map(|(s, t)| t - s)
                .collect();

            deltas.push(output_deltas);

            for index in (0..num_layers - 1).rev() {
                deltas.push(self.0[index].backward_hidden(
                    &all_outputs[index + 1],
                    deltas.last().unwrap(),
                    &self.0[index + 1],
                ));
            }

            deltas.reverse();

            for layer_idx in 0..num_layers {
                for neuron_idx in 0..deltas[layer_idx].len() {
                    let delta = deltas[layer_idx][neuron_idx];
                    accumulated_bias_gradients[layer_idx][neuron_idx] += delta;
                    for input_idx in 0..all_outputs[layer_idx].len() {
                        let input = all_outputs[layer_idx][input_idx];
                        accumulated_weight_gradients[layer_idx][neuron_idx][input_idx] +=
                            delta * input;
                    }
                }
            }
        }

        let batch_size = batch.len() as f64;

        for layer_idx in 0..num_layers {
            for neuron_idx in 0..self.0[layer_idx].neurons.len() {
                for weight_idx in 0..self.0[layer_idx].neurons[neuron_idx].weights.len() {
                    let avg_gradient = accumulated_weight_gradients[layer_idx][neuron_idx]
                        [weight_idx]
                        / batch_size;
                    self.0[layer_idx].neurons[neuron_idx].weights[weight_idx] +=
                        learning_rate * avg_gradient;
                }

                let avg_bias_gradient =
                    accumulated_bias_gradients[layer_idx][neuron_idx] / batch_size;
                self.0[layer_idx].neurons[neuron_idx].biais += learning_rate * avg_bias_gradient;
            }
        }
    }

    fn softmax(outputs: &Vec<f64>) -> Vec<f64> {
        let max = outputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = outputs.iter().map(|&x| (x - max).exp()).collect();
        let sum: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum).collect()
    }

    pub fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n---\n")
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let content = self.to_string();

        fs::write(path, content).map_err(|e| format!("Failed to save network to {}: {}", path, e))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to load network from {}: {}", path, e))?;

        Network::new(content)
    }

    pub fn check_gradients(&self) -> Result<(), String> {
        for (layer_idx, layer) in self.0.iter().enumerate() {
            for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
                for (weight_idx, &weight) in neuron.weights.iter().enumerate() {
                    if weight.is_nan() {
                        return Err(format!(
                            "NaN weight at layer {}, neuron {}, weight {}",
                            layer_idx, neuron_idx, weight_idx
                        ));
                    }
                    if weight.is_infinite() {
                        return Err(format!(
                            "Infinite weight at layer {}, neuron {}, weight {}",
                            layer_idx, neuron_idx, weight_idx
                        ));
                    }
                    if weight.abs() > 1000.0 {
                        eprintln!(
                            "WARNING: Large weight at layer {}, neuron {}, weight {}: {}",
                            layer_idx, neuron_idx, weight_idx, weight
                        );
                    }
                }

                if neuron.biais.is_nan() {
                    return Err(format!(
                        "NaN bias at layer {}, neuron {}",
                        layer_idx, neuron_idx
                    ));
                }
                if neuron.biais.is_infinite() {
                    return Err(format!(
                        "Infinite bias at layer {}, neuron {}",
                        layer_idx, neuron_idx
                    ));
                }
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_architecture(&self) -> Vec<usize> {
        self.0.iter().map(|layer| layer.neurons.len()).collect()
    }

    pub fn count_parameters(&self) -> usize {
        self.0
            .iter()
            .map(|layer| {
                layer
                    .neurons
                    .iter()
                    .map(|neuron| neuron.weights.len() + 1)
                    .sum::<usize>()
            })
            .sum()
    }
}
