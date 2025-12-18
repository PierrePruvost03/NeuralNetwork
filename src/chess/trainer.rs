use crate::chess::config::TrainingConfig;
use crate::chess::fen::FenPosition;
use crate::network::datastruct::network::Network;
use crate::parse_config::Config;
use rand::seq::SliceRandom;
#[allow(deprecated)]
use rand::thread_rng;
use std::fs;

pub fn run_train(config: &Config) -> Result<(), String> {
    println!("=== Training Mode ===\n");

    // Load training configuration
    let train_config = if let Some(ref conf_file) = config.configfile {
        println!("Loading training configuration from '{}'...", conf_file);
        TrainingConfig::load(conf_file)?
    } else {
        println!("Using default training configuration");
        TrainingConfig::default()
    };

    println!("\nTraining Configuration:");
    println!("  Learning rate: {}", train_config.learning_rate);
    println!("  Epochs: {}", train_config.epochs);
    println!("  Batch size: {}", train_config.batch_size);
    println!("  Patience: {}", train_config.patience);
    println!("  Train ratio: {}", train_config.train_ratio);
    println!("  Architecture: {:?}", train_config.hidden_layers);
    if train_config.lr_decay_enabled {
        println!(
            "  LR decay: enabled (rate={}, step={})",
            train_config.lr_decay_rate, train_config.lr_decay_step
        );
    }
    println!();

    let mut network = if std::path::Path::new(&config.loadfile).exists() {
        println!("Loading existing network from '{}'...", config.loadfile);
        Network::load(&config.loadfile)?
    } else {
        println!(
            "Creating new network (file '{}' not found)...",
            config.loadfile
        );
        create_chess_network(&train_config)?
    };

    println!("Reading training data from '{}'...", config.chessfile);
    let raw_data = read_training_file(&config.chessfile)?;
    println!("  Loaded {} training examples", raw_data.len());

    println!("Converting FEN positions to network inputs...");
    let mut training_data = convert_to_training_data(&raw_data)?;
    println!("  Converted {} examples", training_data.len());

    // Shuffle the dataset to ensure random distribution in train/val split
    println!("Shuffling dataset...");
    #[allow(deprecated)]
    let mut rng = thread_rng();
    training_data.shuffle(&mut rng);

    let (train_set, val_set) = split_dataset(&training_data, train_config.train_ratio);
    println!(
        "  Training set: {} examples, Validation set: {} examples",
        train_set.len(),
        val_set.len()
    );

    println!("\nStarting training...");
    train_network(&mut network, &train_set, &val_set, &train_config)?;

    let save_path = config.savefile.as_ref().unwrap_or(&config.loadfile);
    println!("\nSaving trained network to '{}'...", save_path);
    network.save(save_path)?;
    println!("✓ Network saved successfully");

    Ok(())
}

fn create_chess_network(train_config: &TrainingConfig) -> Result<Network, String> {
    let input_size = 832; // 64 cases * 13 états
    let output_size = 3; // Nothing, Check, Checkmate

    let mut layers = train_config.hidden_layers.clone();
    layers.push(output_size);

    println!("  Architecture: {} -> {:?}", input_size, layers);

    let weight_range = (train_config.weight_min, train_config.weight_max);
    let bias_range = (train_config.bias_min, train_config.bias_max);

    Ok(Network::new_random(
        input_size,
        layers,
        weight_range,
        bias_range,
    ))
}

fn read_training_file(path: &str) -> Result<Vec<(String, String)>, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Cannot read file {}: {}", path, e))?;

    let mut data = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 {
            return Err(format!(
                "Invalid format at line {}: expected FEN (6 fields) + LABEL",
                line_num + 1
            ));
        }

        let fen = parts[..6].join(" ");
        let label = parts[6..].join(" ");

        data.push((fen, label));
    }

    if data.is_empty() {
        return Err(format!("No training data found in {}", path));
    }

    Ok(data)
}

fn convert_to_training_data(
    raw_data: &Vec<(String, String)>,
) -> Result<Vec<(Vec<f64>, Vec<f64>)>, String> {
    let mut training_data = Vec::new();

    for (i, (fen, label)) in raw_data.iter().enumerate() {
        let position = FenPosition::parse(fen)
            .map_err(|e| format!("Error parsing FEN at example {}: {}", i + 1, e))?;

        let inputs = position.to_inputs();

        let targets = label_to_targets(label)?;

        training_data.push((inputs, targets));
    }

    Ok(training_data)
}

fn label_to_targets(label: &str) -> Result<Vec<f64>, String> {
    let normalized = if label.contains("Nothing") {
        "Nothing"
    } else if label.contains("Checkmate") {
        "Checkmate"
    } else if label.contains("Check") {
        "Check"
    } else {
        return Err(format!("Unknown label: {}", label));
    };

    match normalized {
        "Nothing" => Ok(vec![1.0, 0.0, 0.0]),
        "Check" => Ok(vec![0.0, 1.0, 0.0]),
        "Checkmate" => Ok(vec![0.0, 0.0, 1.0]),
        _ => Err(format!("Unknown label: {}", label)),
    }
}

fn split_dataset<T: Clone>(data: &Vec<T>, train_ratio: f64) -> (Vec<T>, Vec<T>) {
    let split_idx = (data.len() as f64 * train_ratio) as usize;
    let train_set = data[..split_idx].to_vec();
    let val_set = data[split_idx..].to_vec();
    (train_set, val_set)
}

fn train_network(
    network: &mut Network,
    train_set: &Vec<(Vec<f64>, Vec<f64>)>,
    val_set: &Vec<(Vec<f64>, Vec<f64>)>,
    train_config: &TrainingConfig,
) -> Result<(), String> {
    let epochs = train_config.epochs;
    let patience_limit = train_config.patience;

    let mut best_val_loss = f64::INFINITY;
    let mut patience = 0;

    for epoch in 0..epochs {
        let learning_rate = train_config.get_learning_rate(epoch);
        let mut train_loss = 0.0;

        if train_config.batch_size == 1 {
            for (inputs, targets) in train_set {
                network.train(inputs, targets, learning_rate);
                let outputs = network.exec(inputs.clone());
                let loss = calculate_mse(&outputs, targets);
                train_loss += loss;
            }
        } else {
            for batch_start in (0..train_set.len()).step_by(train_config.batch_size) {
                let batch_end = (batch_start + train_config.batch_size).min(train_set.len());

                let batch = &train_set[batch_start..batch_end];

                network.train_batch(batch, learning_rate);

                for (inputs, targets) in batch {
                    let outputs = network.exec(inputs.clone());
                    let loss = calculate_mse(&outputs, targets);
                    train_loss += loss;
                }
            }
        }
        train_loss /= train_set.len() as f64;

        let mut val_loss = 0.0;
        let mut correct = 0;
        for (inputs, targets) in val_set {
            let outputs = network.exec(inputs.clone());
            let loss = calculate_mse(&outputs, targets);
            val_loss += loss;

            if are_predictions_equal(&outputs, targets) {
                correct += 1;
            }
        }
        val_loss /= val_set.len() as f64;
        let val_accuracy = (correct as f64 / val_set.len() as f64) * 100.0;

        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            patience = 0;
        } else {
            patience += 1;
            if patience >= patience_limit {
                println!(
                    "Epoch {}: train_loss={:.6}, val_loss={:.6}, val_acc={:.2}%, lr={:.6}",
                    epoch, train_loss, val_loss, val_accuracy, learning_rate
                );
                println!("Early stopping at epoch {} (patience reached)", epoch);
                break;
            }
        }

        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}: train_loss={:.6}, val_loss={:.6}, val_acc={:.2}%, lr={:.6}",
                epoch, train_loss, val_loss, val_accuracy, learning_rate
            );
        }
    }

    println!("\nTraining completed!");
    println!("  Best validation loss: {:.6}", best_val_loss);

    Ok(())
}

fn calculate_mse(outputs: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    outputs
        .iter()
        .zip(targets.iter())
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f64>()
        / outputs.len() as f64
}

fn are_predictions_equal(outputs: &Vec<f64>, targets: &Vec<f64>) -> bool {
    let predicted_class = outputs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let target_class = targets
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    predicted_class == target_class
}
