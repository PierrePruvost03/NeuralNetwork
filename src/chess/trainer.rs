use crate::chess::config::TrainingConfig;
use crate::chess::fen::FenPosition;
use crate::network::datastruct::network::Network;
use crate::parse_config::Config;
use rand::seq::SliceRandom;
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
        let mut net = Network::load(&config.loadfile)?;

        let dropout_rates = vec![0.3, 0.2, 0.1, 0.05, 0.0];
        println!("  Reconfiguring dropout rates: {:?}", dropout_rates);
        for (idx, layer) in net.0.iter_mut().enumerate() {
            if idx < dropout_rates.len() {
                layer.set_dropout(dropout_rates[idx]);
            }
        }

        net
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
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    training_data.shuffle(&mut rng);

    let (train_set, val_set) = split_dataset(&training_data, train_config.train_ratio);
    println!(
        "  Training set: {} examples, Validation set: {} examples",
        train_set.len(),
        val_set.len()
    );

    println!("\nStarting training with Softmax + Cross-Entropy...");
    train_network_softmax(&mut network, &train_set, &val_set, &train_config)?;

    let save_path = config.savefile.as_ref().unwrap_or(&config.loadfile);
    println!("\nSaving trained network to '{}'...", save_path);
    network.save(save_path)?;
    println!("✓ Network saved successfully");

    Ok(())
}

fn create_chess_network(train_config: &TrainingConfig) -> Result<Network, String> {
    let input_size = 833; // 64 cases * 13 états + 1 (active_color)
    let output_size = 5; // Nothing, Check White, Check Black, Checkmate White, Checkmate Black

    let mut layers = train_config.hidden_layers.clone();
    layers.push(output_size);

    println!("  Architecture: {} -> {:?}", input_size, layers);

    let dropout_rates = vec![0.4, 0.3, 0.2, 0.1, 0.0];

    println!("  Using He initialization with ReLU activation");
    println!("  Dropout rates: {:?}", dropout_rates);

    let network = Network::new_random_he(
        input_size as u32,
        layers.iter().map(|&x| x as u32).collect(),
        dropout_rates,
        "linear",
    );

    println!("  Total parameters: {}", network.count_parameters());

    Ok(network)
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
    // Pour 5 classes: Nothing, Check White, Check Black, Checkmate White, Checkmate Black
    if label.contains("Nothing") {
        Ok(vec![1.0, 0.0, 0.0, 0.0, 0.0])
    } else if label.contains("Checkmate White") {
        Ok(vec![0.0, 0.0, 0.0, 1.0, 0.0])
    } else if label.contains("Checkmate Black") {
        Ok(vec![0.0, 0.0, 0.0, 0.0, 1.0])
    } else if label.contains("Check White") {
        Ok(vec![0.0, 1.0, 0.0, 0.0, 0.0])
    } else if label.contains("Check Black") {
        Ok(vec![0.0, 0.0, 1.0, 0.0, 0.0])
    } else {
        Err(format!("Unknown label: {}", label))
    }
}

fn split_dataset<T: Clone>(data: &Vec<T>, train_ratio: f64) -> (Vec<T>, Vec<T>) {
    let split_idx = (data.len() as f64 * train_ratio) as usize;
    let train_set = data[..split_idx].to_vec();
    let val_set = data[split_idx..].to_vec();
    (train_set, val_set)
}

fn train_network_softmax(
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

        let mut shuffled_train_set = train_set.clone();
        let mut rng = thread_rng();
        shuffled_train_set.shuffle(&mut rng);

        network.set_training_mode(true);

        if train_config.batch_size == 1 {
            for (inputs, targets) in &shuffled_train_set {
                network.train_softmax_ce(inputs, targets, learning_rate);
                let outputs = network.exec(inputs.clone());
                let loss = calculate_cross_entropy(&outputs, targets);
                train_loss += loss;
            }
        } else {
            for batch_start in (0..shuffled_train_set.len()).step_by(train_config.batch_size) {
                let batch_end =
                    (batch_start + train_config.batch_size).min(shuffled_train_set.len());

                let batch = &shuffled_train_set[batch_start..batch_end].to_vec();

                network.train_batch_softmax_ce(batch, learning_rate);

                for (inputs, targets) in batch {
                    let outputs = network.exec(inputs.clone());
                    let loss = calculate_cross_entropy(&outputs, targets);
                    train_loss += loss;
                }
            }
        }
        train_loss /= shuffled_train_set.len() as f64;

        if let Err(e) = network.check_gradients() {
            eprintln!("Gradient check failed at epoch {}: {}", epoch, e);
            return Err(format!("Training diverged at epoch {}: {}", epoch, e));
        }

        network.set_training_mode(false);

        let mut val_loss = 0.0;
        let mut correct = 0;
        for (inputs, targets) in val_set {
            let outputs = network.exec(inputs.clone());
            let loss = calculate_cross_entropy(&outputs, targets);
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

        println!(
            "Epoch {}: train_loss={:.6}, val_loss={:.6}, val_acc={:.2}%, lr={:.6}",
            epoch, train_loss, val_loss, val_accuracy, learning_rate
        );
    }

    Ok(())
}

fn softmax(outputs: &Vec<f64>) -> Vec<f64> {
    let max = outputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_values: Vec<f64> = outputs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exp_values.iter().sum();
    exp_values.iter().map(|&x| x / sum).collect()
}

fn calculate_cross_entropy(outputs: &Vec<f64>, targets: &Vec<f64>) -> f64 {
    let epsilon = 1e-15;

    let softmax_outputs = softmax(outputs);

    softmax_outputs
        .iter()
        .zip(targets.iter())
        .map(|(o, t)| {
            let o_clipped = o.clamp(epsilon, 1.0 - epsilon);
            -t * o_clipped.ln()
        })
        .sum::<f64>()
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
