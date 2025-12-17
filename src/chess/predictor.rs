use crate::parse_config::Config;

pub fn run_predict(config: &Config) -> Result<(), String> {
    println!("Running in PREDICT mode...");
    println!("  Load file: {}", config.loadfile);
    println!("  Chess file: {}", config.chessfile);

    // TODO: Implémenter la logique de prédiction

    Ok(())
}
