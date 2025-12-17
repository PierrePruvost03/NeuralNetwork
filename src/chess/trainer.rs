use crate::parse_config::Config;

pub fn run_train(config: &Config) -> Result<(), String> {
    println!("Running in TRAIN mode...");
    println!("  Load file: {}", config.loadfile);
    println!("  Chess file: {}", config.chessfile);

    if let Some(ref save) = config.savefile {
        println!("  Save file: {}", save);
    } else {
        println!("  Save file: {} (default)", config.loadfile);
    }

    // TODO: Implémenter la logique de training

    Ok(())
}
