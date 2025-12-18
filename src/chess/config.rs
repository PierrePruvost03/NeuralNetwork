use std::fs;

#[derive(Debug, Clone)]
pub struct TrainingConfig {
    // Hyperparamètres d'apprentissage
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
    pub patience: usize,

    // Ratio de split train/validation
    pub train_ratio: f64,

    // Architecture du réseau
    pub hidden_layers: Vec<u32>,

    // Initialisation des poids
    pub weight_min: f64,
    pub weight_max: f64,
    pub bias_min: f64,
    pub bias_max: f64,

    // Learning rate decay
    pub lr_decay_enabled: bool,
    pub lr_decay_rate: f64,
    pub lr_decay_step: usize,
}

impl TrainingConfig {
    /// Charge une config depuis un fichier
    pub fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Cannot read config file {}: {}", path, e))?;

        Self::parse(&content)
    }

    pub fn parse(content: &str) -> Result<Self, String> {
        let mut config = Self::default();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 {
                return Err(format!(
                    "Invalid format at line {}: expected KEY = VALUE",
                    line_num + 1
                ));
            }

            let key = parts[0].trim();
            let value = parts[1].trim();

            match key {
                "learning_rate" => {
                    config.learning_rate = value
                        .parse()
                        .map_err(|_| format!("Invalid learning_rate: {}", value))?;
                }
                "epochs" => {
                    config.epochs = value
                        .parse()
                        .map_err(|_| format!("Invalid epochs: {}", value))?;
                }
                "batch_size" => {
                    config.batch_size = value
                        .parse()
                        .map_err(|_| format!("Invalid batch_size: {}", value))?;
                }
                "patience" => {
                    config.patience = value
                        .parse()
                        .map_err(|_| format!("Invalid patience: {}", value))?;
                }
                "train_ratio" => {
                    config.train_ratio = value
                        .parse()
                        .map_err(|_| format!("Invalid train_ratio: {}", value))?;
                }
                "hidden_layers" => {
                    config.hidden_layers = Self::parse_vec_u32(value)?;
                }
                "weight_min" => {
                    config.weight_min = value
                        .parse()
                        .map_err(|_| format!("Invalid weight_min: {}", value))?;
                }
                "weight_max" => {
                    config.weight_max = value
                        .parse()
                        .map_err(|_| format!("Invalid weight_max: {}", value))?;
                }
                "bias_min" => {
                    config.bias_min = value
                        .parse()
                        .map_err(|_| format!("Invalid bias_min: {}", value))?;
                }
                "bias_max" => {
                    config.bias_max = value
                        .parse()
                        .map_err(|_| format!("Invalid bias_max: {}", value))?;
                }
                "lr_decay_enabled" => {
                    config.lr_decay_enabled = Self::parse_bool(value)?;
                }
                "lr_decay_rate" => {
                    config.lr_decay_rate = value
                        .parse()
                        .map_err(|_| format!("Invalid lr_decay_rate: {}", value))?;
                }
                "lr_decay_step" => {
                    config.lr_decay_step = value
                        .parse()
                        .map_err(|_| format!("Invalid lr_decay_step: {}", value))?;
                }
                _ => {
                    return Err(format!("Unknown configuration key: {}", key));
                }
            }
        }

        config.validate()?;

        Ok(config)
    }

    pub fn default() -> Self {
        TrainingConfig {
            learning_rate: 0.01,
            epochs: 1000,
            batch_size: 1,
            patience: 50,
            train_ratio: 0.8,
            hidden_layers: vec![256, 128, 64],
            weight_min: -0.3,
            weight_max: 0.3,
            bias_min: -0.1,
            bias_max: 0.1,
            lr_decay_enabled: false,
            lr_decay_rate: 0.95,
            lr_decay_step: 100,
        }
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let content = self.to_string();
        fs::write(path, content).map_err(|e| format!("Cannot write config file {}: {}", path, e))
    }

    pub fn to_string(&self) -> String {
        format!(
            "# Training Configuration\n\
            # Learning parameters\n\
            learning_rate = {}\n\
            epochs = {}\n\
            batch_size = {}\n\
            patience = {}\n\
            \n\
            # Data split\n\
            train_ratio = {}\n\
            \n\
            # Network architecture\n\
            hidden_layers = {}\n\
            \n\
            # Weight initialization\n\
            weight_min = {}\n\
            weight_max = {}\n\
            bias_min = {}\n\
            bias_max = {}\n\
            \n\
            # Learning rate decay\n\
            lr_decay_enabled = {}\n\
            lr_decay_rate = {}\n\
            lr_decay_step = {}\n",
            self.learning_rate,
            self.epochs,
            self.batch_size,
            self.patience,
            self.train_ratio,
            self.format_vec_u32(&self.hidden_layers),
            self.weight_min,
            self.weight_max,
            self.bias_min,
            self.bias_max,
            self.lr_decay_enabled,
            self.lr_decay_rate,
            self.lr_decay_step,
        )
    }

    fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(format!(
                "Invalid learning_rate: {} (must be 0 < lr <= 1)",
                self.learning_rate
            ));
        }

        if self.epochs == 0 {
            return Err(String::from("epochs must be > 0"));
        }

        if self.batch_size == 0 {
            return Err(String::from("batch_size must be > 0"));
        }

        if self.patience == 0 {
            return Err(String::from("patience must be > 0"));
        }

        if self.train_ratio <= 0.0 || self.train_ratio >= 1.0 {
            return Err(format!(
                "Invalid train_ratio: {} (must be 0 < ratio < 1)",
                self.train_ratio
            ));
        }

        if self.hidden_layers.is_empty() {
            return Err(String::from("hidden_layers cannot be empty"));
        }

        if self.weight_min >= self.weight_max {
            return Err(String::from("weight_min must be < weight_max"));
        }

        if self.bias_min >= self.bias_max {
            return Err(String::from("bias_min must be < bias_max"));
        }

        if self.lr_decay_rate <= 0.0 || self.lr_decay_rate >= 1.0 {
            return Err(format!(
                "Invalid lr_decay_rate: {} (must be 0 < rate < 1)",
                self.lr_decay_rate
            ));
        }

        if self.lr_decay_step == 0 {
            return Err(String::from("lr_decay_step must be > 0"));
        }

        Ok(())
    }

    /// Format : "[256, 128, 64]" ou "256, 128, 64" ou "256 128 64"
    fn parse_vec_u32(s: &str) -> Result<Vec<u32>, String> {
        let s = s.trim();

        let s = s.trim_start_matches('[').trim_end_matches(']');

        let numbers: Result<Vec<u32>, _> = s
            .split(|c| c == ',' || c == ' ')
            .filter(|x| !x.is_empty())
            .map(|x| x.trim().parse())
            .collect();

        numbers.map_err(|_| format!("Invalid hidden_layers format: {}", s))
    }

    fn format_vec_u32(&self, vec: &Vec<u32>) -> String {
        format!(
            "[{}]",
            vec.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn parse_bool(s: &str) -> Result<bool, String> {
        match s.to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            _ => Err(format!("Invalid boolean value: {}", s)),
        }
    }

    pub fn get_learning_rate(&self, epoch: usize) -> f64 {
        if self.lr_decay_enabled {
            let decay_count = epoch / self.lr_decay_step;
            self.learning_rate * self.lr_decay_rate.powi(decay_count as i32)
        } else {
            self.learning_rate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.epochs, 1000);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_parse_simple_config() {
        let content = "learning_rate = 0.05\nepochs = 500\nbatch_size = 32";
        let config = TrainingConfig::parse(content).unwrap();
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.epochs, 500);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_parse_with_comments() {
        let content = "# Comment\nlearning_rate = 0.01\n// Another comment\nepochs = 100";
        let config = TrainingConfig::parse(content).unwrap();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.epochs, 100);
    }

    #[test]
    fn test_parse_hidden_layers() {
        assert_eq!(
            TrainingConfig::parse_vec_u32("[256, 128, 64]").unwrap(),
            vec![256, 128, 64]
        );
        assert_eq!(
            TrainingConfig::parse_vec_u32("256, 128, 64").unwrap(),
            vec![256, 128, 64]
        );
        assert_eq!(
            TrainingConfig::parse_vec_u32("256 128 64").unwrap(),
            vec![256, 128, 64]
        );
    }

    #[test]
    fn test_parse_bool() {
        assert_eq!(TrainingConfig::parse_bool("true").unwrap(), true);
        assert_eq!(TrainingConfig::parse_bool("false").unwrap(), false);
        assert_eq!(TrainingConfig::parse_bool("1").unwrap(), true);
        assert_eq!(TrainingConfig::parse_bool("0").unwrap(), false);
        assert_eq!(TrainingConfig::parse_bool("yes").unwrap(), true);
        assert_eq!(TrainingConfig::parse_bool("no").unwrap(), false);
    }

    #[test]
    fn test_lr_decay() {
        let mut config = TrainingConfig::default();
        config.lr_decay_enabled = true;
        config.learning_rate = 0.1;
        config.lr_decay_rate = 0.9;
        config.lr_decay_step = 100;

        assert!((config.get_learning_rate(0) - 0.1).abs() < 1e-10);
        assert!((config.get_learning_rate(50) - 0.1).abs() < 1e-10);
        assert!((config.get_learning_rate(100) - 0.09).abs() < 1e-10);
        assert!((config.get_learning_rate(200) - 0.081).abs() < 1e-10);
    }

    #[test]
    fn test_validation() {
        let mut config = TrainingConfig::default();

        config.learning_rate = -0.1;
        assert!(config.validate().is_err());

        config.learning_rate = 0.01;
        config.epochs = 0;
        assert!(config.validate().is_err());

        config.epochs = 100;
        config.train_ratio = 1.5;
        assert!(config.validate().is_err());
    }
}
