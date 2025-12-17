use std::fs;
use std::io::Write;
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

    pub fn new_random(
        mut nb_input: u32,
        nb_perceptron: Vec<u32>,
        w_range: (f64, f64),
        b_range: (f64, f64),
    ) -> Network {
        Network(
            nb_perceptron
                .iter()
                .map(|&nb| {
                    let r = Layer::new_random(nb, nb_input, &w_range, &b_range);
                    nb_input = nb;
                    r
                })
                .collect(),
        )
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
}
