use my_torch_analyzer::chess::config::TrainingConfig;
use my_torch_analyzer::network::datastruct::network::Network;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        print_help();
        std::process::exit(0);
    }

    if args.len() < 3 {
        eprintln!("Error: Not enough arguments");
        eprintln!();
        print_help();
        std::process::exit(84);
    }

    if (args.len() - 1) % 2 != 0 {
        eprintln!("Error: Arguments must come in pairs (config_file nb)");
        eprintln!();
        print_help();
        std::process::exit(84);
    }

    println!("=== MY_TORCH Network Generator ===\n");

    let mut total_generated = 0;

    let mut i = 1;
    while i < args.len() {
        let config_file = &args[i];
        let nb_str = &args[i + 1];

        let nb: usize = match nb_str.parse() {
            Ok(n) => {
                if n == 0 {
                    eprintln!("Error: Number of networks must be > 0");
                    std::process::exit(84);
                }
                n
            }
            Err(_) => {
                eprintln!("Error: Invalid number '{}' for {}", nb_str, config_file);
                std::process::exit(84);
            }
        };

        println!("Processing: {} ({} networks)", config_file, nb);

        match generate_networks(config_file, nb) {
            Ok(count) => {
                println!("  ✓ Generated {} networks", count);
                total_generated += count;
            }
            Err(e) => {
                eprintln!("  ❌ Error: {}", e);
                std::process::exit(84);
            }
        }

        i += 2;
    }

    println!("\n✅ Total networks generated: {}", total_generated);
}

/// Generate N networks from a configuration file
fn generate_networks(config_file: &str, nb: usize) -> Result<usize, String> {
    // Load configuration
    let config = TrainingConfig::load(config_file)
        .map_err(|e| format!("Failed to load config {}: {}", config_file, e))?;

    let base_name = extract_base_name(config_file);

    let input_size = 833; // 64 squares × 13 states + 1 (active_color)

    let output_size = 5; // Nothing, Check White, Check Black, Checkmate White, Checkmate Black

    let mut layers = config.hidden_layers.clone();
    layers.push(output_size);

    // Get dropout rates from configuration
    let dropout_rates = config.get_dropout_rates();
    // Use 'linear' activation for output layer (softmax is applied at network level)
    let output_activation = "linear";

    // Validate that dropout_rates length matches number of layers
    if dropout_rates.len() != layers.len() {
        return Err(format!(
            "Dropout rates count ({}) must match number of layers ({})",
            dropout_rates.len(),
            layers.len()
        ));
    }

    for i in 1..=nb {
        let network = Network::new_random_he(
            input_size,
            layers.clone(),
            dropout_rates.clone(),
            output_activation,
        );

        let filename = if nb == 1 {
            format!("{}.nn", base_name)
        } else {
            format!("{}_{}.nn", base_name, i)
        };

        network
            .save(&filename)
            .map_err(|e| format!("Failed to save {}: {}", filename, e))?;

        if nb <= 10 {
            println!("    Created: {}", filename);
        }
    }

    Ok(nb)
}

fn extract_base_name(path: &str) -> String {
    let filename = path.rsplit('/').next().unwrap_or(path);
    let name = filename.split('.').next().unwrap_or(filename);

    name.to_string()
}

fn print_help() {
    println!("USAGE");
    println!("    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]");
    println!();
    println!("DESCRIPTION");
    println!("    config_file_i  Configuration file containing description of a neural network");
    println!("                   to generate. Uses .conf files from configs/ directory.");
    println!();
    println!("    nb_i           Number of neural networks to generate based on the");
    println!("                   configuration file.");
    println!();
    println!("EXAMPLES");
    println!("    # Generate 1 network from default config");
    println!("    ./my_torch_generator configs/default.conf 1");
    println!();
    println!("    # Generate 3 networks from default config");
    println!("    ./my_torch_generator configs/default.conf 3");
    println!();
    println!("    # Generate from multiple configs");
    println!("    ./my_torch_generator configs/fast.conf 2 configs/deep.conf 1");
    println!();
    println!("    Example:");
    println!("    $ ./my_torch_generator configs/default.conf 3");
    println!("    Creates: default_1.nn, default_2.nn, default_3.nn");
}
