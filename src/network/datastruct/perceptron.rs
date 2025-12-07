use rand::distr::uniform::Error;
use std::str::FromStr;

pub fn function_getter(key: String) -> Result<fn(f64, f64) -> f64, String> {
    match key.as_str() {
        "sigmoid" => Ok(|i, b| 1. / (1. + (i + b).exp())),
        _ => Err(format!("unknow function '{}'", key)),
    }
}

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
