// use rand::Rng;

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
//         Perceptron {
//             func: |per, sum| (sum * per.learing_rate) >= 0.,
//             weigth: (0..nb_weigth)
//                 .map(|_| rand::random_range(0.0..1.))
//                 .collect(),
//             learing_rate: learning_rate,
//             biais: biais,
//         }
//     }
// }

mod network;
use crate::network::datastruct::perceptron::Perceptron;

fn main() {
    let les_pieds = Perceptron::new(String::from("sigmoid 0. 1 2")).unwrap();
    println!("{:?}", les_pieds.exec(vec![0., 0.]));
    println!("{}", les_pieds.to_string());
}
