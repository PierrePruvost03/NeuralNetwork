use crate::network::{
    self,
    datastruct::perceptron::{self, Perceptron},
};

pub fn exec_layer(inputs: Vec<f64>, layer: Vec<Perceptron>) -> Vec<f64> {
    layer.iter().map(|p| p.exec(&inputs)).collect()
}

pub fn exec_network(inputs: Vec<f64>, mut network: Vec<Vec<Perceptron>>) -> Vec<f64> {
    if network.len() == 0 {
        return inputs;
    }
    let current = network.remove(0);
    exec_network(exec_layer(inputs, current), network)
}
