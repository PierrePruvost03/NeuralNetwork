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

    pub fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n---\n")
    }
}
