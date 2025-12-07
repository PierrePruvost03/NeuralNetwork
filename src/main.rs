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
use crate::network::{
    datastruct::perceptron::Perceptron,
    layer::{exec_network, generate_random_network},
};

fn main() {
    let network = generate_random_network(3, vec![4, 3], (0., 3.4), (-1., 1.5));

    println!("{:?}", exec_network(vec![1., 2., 3.], network));

    let les_pieds = Perceptron::new(String::from("sigmoid 0. 1 2")).unwrap();
    let inputs = vec![0., 0.];
    println!("{:?}", les_pieds.exec(&inputs));
    println!("{}", les_pieds.to_string());
}
