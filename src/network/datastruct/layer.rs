use crate::network::datastruct::perceptron::{get_derivative, Perceptron};
use rand::thread_rng;
use rand::Rng;

pub struct Layer {
    pub neurons: Vec<Perceptron>,
    pub dropout_rate: f64,
    pub training_mode: bool,
}

impl Layer {
    #[allow(dead_code)]
    pub fn neurons(&self) -> &Vec<Perceptron> {
        &self.neurons
    }

    #[allow(dead_code)]
    pub fn neurons_mut(&mut self) -> &mut Vec<Perceptron> {
        &mut self.neurons
    }
}

impl Layer {
    pub fn new(config: String) -> Result<Self, String> {
        Ok(Layer {
            neurons: config
                .split("\n")
                .map(|line| Perceptron::new(String::from(line)).unwrap())
                .collect(),
            dropout_rate: 0.0,
            training_mode: false,
        })
    }

    #[allow(dead_code)]
    pub fn new_random_with_activation(
        nb_perceptron: u32,
        nb_weight: u32,
        w_range: &(f64, f64),
        b_range: &(f64, f64),
        activation: &str,
    ) -> Layer {
        Layer {
            neurons: (0..nb_perceptron)
                .map(|_| {
                    Perceptron::new_random_with_activation(nb_weight, w_range, b_range, activation)
                })
                .collect(),
            dropout_rate: 0.0,
            training_mode: false,
        }
    }

    pub fn new_random_he(nb_perceptron: u32, nb_weight: u32, activation: &str) -> Layer {
        Layer {
            neurons: (0..nb_perceptron)
                .map(|_| Perceptron::new_random_he(nb_weight, activation))
                .collect(),
            dropout_rate: 0.0,
            training_mode: false,
        }
    }

    pub fn set_dropout(&mut self, rate: f64) {
        self.dropout_rate = rate;
    }

    pub fn set_training_mode(&mut self, training: bool) {
        self.training_mode = training;
    }

    pub fn exec(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.neurons.iter().map(|p| p.exec(&inputs)).collect()
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let outputs: Vec<f64> = self.neurons.iter().map(|p| p.exec(&inputs)).collect();

        if self.training_mode && self.dropout_rate > 0.0 {
            let mut rng = thread_rng();
            let keep_prob = 1.0 - self.dropout_rate;

            outputs
                .iter()
                .map(|&output| {
                    if rng.gen::<f64>() < keep_prob {
                        output / keep_prob
                    } else {
                        0.0
                    }
                })
                .collect()
        } else {
            outputs
        }
    }

    #[allow(dead_code)]
    pub fn backward_output(&self, outputs: &Vec<f64>, targets: &Vec<f64>) -> Vec<f64> {
        let mut deltas: Vec<f64> = vec![];

        for index in 0..outputs.len() {
            let derivative = get_derivative(&self.neurons[index].func_id);
            let output_error = targets[index] - outputs[index];
            let output_delta = output_error * derivative(outputs[index]);
            deltas.push(output_delta);
        }
        deltas
    }

    pub fn backward_hidden(
        &self,
        outputs: &Vec<f64>,
        next_deltas: &Vec<f64>,
        next_layer: &Layer,
    ) -> Vec<f64> {
        let mut deltas: Vec<f64> = vec![];
        for i in 0..self.neurons.len() {
            let derivative = get_derivative(&self.neurons[i].func_id);
            let mut error_sum = 0.;
            for j in 0..next_layer.neurons.len() {
                error_sum += next_layer.neurons[j].weights[i] * next_deltas[j];
            }
            deltas.push(error_sum * derivative(outputs[i]));
        }
        deltas
    }

    pub fn update_weights(&mut self, deltas: &Vec<f64>, inputs: &Vec<f64>, learning_rate: f64) {
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons[i].weights.len() {
                self.neurons[i].weights[j] += learning_rate * deltas[i] * inputs[j];
            }
            self.neurons[i].biais += learning_rate * deltas[i];
        }
    }

    pub fn to_string(&self) -> String {
        self.neurons
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
