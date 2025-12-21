use rand::thread_rng;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::str::FromStr;

pub fn function_getter(key: String) -> Result<fn(f64, f64) -> f64, String> {
    match key.as_str() {
        "sigmoid" => Ok(|i, b| 1. / (1. + (-(i + b)).exp())),
        "relu" => Ok(|i, b| {
            let x = i + b;
            if x > 0.0 {
                x
            } else {
                0.0
            }
        }),
        "tanh" => Ok(|i, b| (i + b).tanh()),
        "linear" => Ok(|i, b| i + b),
        _ => Err(format!("unknow function '{}'", key)),
    }
}

pub fn sigmoid_derivate(output: f64) -> f64 {
    output * (1.0 - output)
}

pub fn relu_derivate(output: f64) -> f64 {
    if output > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn tanh_derivate(output: f64) -> f64 {
    return 1.0 - output * output;
}

pub fn linear_derivate(_output: f64) -> f64 {
    return 1.0;
}

pub fn get_derivative(func_id: &str) -> fn(f64) -> f64 {
    match func_id {
        "sigmoid" => sigmoid_derivate,
        "relu" => relu_derivate,
        "tanh" => tanh_derivate,
        "linear" => linear_derivate,
        _ => sigmoid_derivate,
    }
}
#[derive(Debug)]
pub struct Perceptron {
    pub func: fn(f64 /* Input sum */, f64 /* Bias */) -> f64,
    pub func_id: String,
    pub weights: Vec<f64>,
    pub biais: f64,
}

impl Perceptron {
    pub fn new(config: String) -> Result<Self, String> {
        let mut tokens = config.split_whitespace();
        let f_id = String::from(tokens.next().unwrap());
        let f = function_getter(f_id.clone())?;
        let b: f64 = tokens.next().unwrap().parse().unwrap();
        let w: Vec<f64> = tokens.map(f64::from_str).map(Result::unwrap).collect();
        Ok(Perceptron {
            func: f,
            func_id: f_id,
            weights: w,
            biais: b,
        })
    }

    #[allow(dead_code)]
    pub fn new_random_with_activation(
        nb_weight: u32,
        w_range: &(f64, f64),
        b_range: &(f64, f64),
        activation: &str,
    ) -> Self {
        let func_id = String::from(activation);
        let func = function_getter(func_id.clone()).unwrap();
        Perceptron {
            func: func,
            func_id: func_id,
            weights: (0..nb_weight)
                .map(|_| {
                    let mut rng = thread_rng();
                    rng.gen_range(w_range.0..w_range.1)
                })
                .collect(),
            biais: {
                let mut rng = thread_rng();
                rng.gen_range(b_range.0..b_range.1)
            },
        }
    }

    pub fn new_random_he(nb_weight: u32, activation: &str) -> Self {
        let mut rng = thread_rng();

        let std_dev = if activation == "relu" {
            (2.0 / nb_weight as f64).sqrt()
        } else {
            (1.0 / nb_weight as f64).sqrt()
        };

        let normal = Normal::new(0.0, std_dev).unwrap();
        let func_id = String::from(activation);
        let func = function_getter(func_id.clone()).unwrap();

        Perceptron {
            func: func,
            func_id: func_id,
            weights: (0..nb_weight).map(|_| normal.sample(&mut rng)).collect(),
            biais: 0.0,
        }
    }

    pub fn exec(&self, inputs: &Vec<f64>) -> f64 {
        if inputs.len() != self.weights.len() {
            panic!();
        }
        let sum = inputs
            .iter()
            .zip(self.weights.iter())
            .fold(0.0, |acc, (i, w)| acc + i * w);
        (self.func)(sum, self.biais)
    }

    pub fn to_string(&self) -> String {
        let weight_string = self
            .weights
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        format!("{} {} {}", self.func_id, self.biais, weight_string)
    }
}
