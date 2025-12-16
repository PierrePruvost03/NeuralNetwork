use std::ops::Index;

use rand::distr::Iter;

use crate::network::{
    self,
    datastruct::perceptron::{self, Perceptron, sigmoid_derivate},
};

pub struct Layer(Vec<Perceptron>);

impl Layer {
    pub fn new(config: String) -> Result<Self, String> {
        Ok(Layer(
            config
                .split("\n")
                .map(|line| Perceptron::new(String::from(line)).unwrap())
                .collect(),
        ))
    }

    pub fn new_random(
        nb_perceptron: u32,
        nb_weight: u32,
        w_range: &(f64, f64),
        b_range: &(f64, f64),
    ) -> Layer {
        Layer(
            (0..nb_perceptron)
                .map(|_| Perceptron::new_random(nb_weight, w_range, b_range))
                .collect(),
        )
    }

    pub fn exec(&self, inputs: Vec<f64>) -> Vec<f64> {
        self.0.iter().map(|p| p.exec(&inputs)).collect()
    }

    pub fn forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        self.0.iter().map(|p| p.exec(&inputs)).collect()
    }

    pub fn backward_output(
        &self, outputs: &Vec<f64>, targets: &Vec<f64>
    ) -> Vec<f64> {
        let mut deltas: Vec<f64> = vec![];
    
        for index in 0..outputs.len() {
            let output_error = targets[index] - outputs[index];
            let output_delta = output_error * sigmoid_derivate(outputs[index]);
            deltas.push(output_delta);
        }
        deltas
    }

    pub fn backward_hidden(
        &self,
        outputs: &Vec<f64>,
        next_deltas: &Vec<f64>,
        next_layer: &Layer
    ) -> Vec<f64> {
        let mut deltas: Vec<f64> = vec![];
        for i in 0..self.0.len() {
            let mut error_sum = 0.;
            for j in 0..next_layer.0.len() {
                error_sum += next_layer.0[j].weights[i] * next_deltas[j];
            }
            deltas.push(error_sum * sigmoid_derivate(outputs[i]));
        }
        return deltas;
    }

    pub fn update_weights(
        &mut self,
        deltas: &Vec<f64>,
        inputs: &Vec<f64>,
        learning_rate: f64,
    ) {
    for i in 0..self.0.len() {
        for j in 0..self.0[i].weights.len() {
            self.0[i].weights[j] += learning_rate * deltas[i] * inputs[j];
        }
        self.0[i].biais += learning_rate * deltas[i];
        }  
    }

    pub fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
