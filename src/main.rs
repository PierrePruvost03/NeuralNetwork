//

// pub struct Perceptron {
//     func: fn(&Perceptron, f64) -> bool,
//     weigth: Vec<f64>,
//     learing_rate: f64,
//     biais: f64,
// }

// impl Perceptron {
//     fn exec(&self, args: Vec<f64>) -> bool {
//         (self.func)(self, args.iter().sum())
//     }
// }

// impl Perceptron {
//     pub fn new(base_weigths: Vec<f64>, learning_rate: f64, biais: f64) -> Self {
//         Perceptron {
//             func: |per, sum| (sum * per.learing_rate) >= 0.,
//             weigth: base_weigths,
//             learing_rate: learning_rate,
//             biais: biais,
//         }
//     }

//     pub fn new_r(nb_weigth: u32, learning_rate: f64, biais: f64) -> Self {
//
//     }
// }

mod network;
use crate::network::datastruct::{network::Network, perceptron::Perceptron};

fn main() {
    let network = Network::new_random(3, vec![4, 3], (0., 3.4), (-1., 1.5));

    let network_config = network.to_string();

    let parsed_network = Network::new(network_config).unwrap();

    println!("{:?}", network.exec(vec![0., 0., 0.1]));

    println!("{:?}", parsed_network.exec(vec![0., 0., 0.1]));
}
