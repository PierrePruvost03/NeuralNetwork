mod chess;
mod network;
mod parse_config;

use parse_config::{Config, Mode};

fn main() {
    // Parser les arguments CLI
    let config = match Config::parse() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error: {}", e);
            eprintln!();
            Config::print_help();
            std::process::exit(84);
        }
    };

    // Exécuter selon le mode
    let result = match config.mode {
        Mode::Predict => chess::predictor::run_predict(&config),
        Mode::Train => chess::trainer::run_train(&config),
    };

    // Gérer les erreurs
    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(84);
    }
}
