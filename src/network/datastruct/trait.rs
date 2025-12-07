trait NetworkElement {
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

    pub fn new_random(nb_weight: u32, w_range: &(f64, f64), b_range: &(f64, f64)) -> Self;

    pub fn exec<T>(&self, inputs: &Vec<T>) -> T;
    pub fn to_string(&self) -> String;
}
