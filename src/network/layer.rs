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

pub fn generate_random_layer(
    nb_perceptron: u32,
    nb_weight: u32,
    w_range: &(f64, f64),
    b_range: &(f64, f64),
) -> Vec<Perceptron> {
    (0..nb_perceptron)
        .map(|_| Perceptron::new_random(nb_weight, w_range, b_range))
        .collect()
}

pub fn generate_random_network(
    mut nb_input: u32,
    nb_perceptron: Vec<u32>,
    w_range: (f64, f64),
    b_range: (f64, f64),
) -> Vec<Vec<Perceptron>> {
    nb_perceptron
        .iter()
        .map(|&nb| {
            let r = generate_random_layer(nb, nb_input, &w_range, &b_range);
            nb_input = nb;
            r
        })
        .collect()
}
