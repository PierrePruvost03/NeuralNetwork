use crate::network::{
    self,
    datastruct::perceptron::{self, Perceptron},
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

    pub fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
