use crate::chess::fen::FenPosition;
use crate::network::datastruct::network::Network;
use crate::parse_config::Config;
use std::fs;

pub fn run_predict(config: &Config) -> Result<(), String> {
    let network =
        Network::load(&config.loadfile).map_err(|e| format!("Failed to load network: {}", e))?;

    let positions = read_chess_file(&config.chessfile)?;

    for fen_line in positions {
        let fen = extract_fen(&fen_line);

        let position =
            FenPosition::parse(&fen).map_err(|e| format!("Invalid FEN '{}': {}", fen, e))?;

        let inputs = position.to_inputs();

        let outputs = network.exec(inputs);

        let prediction = outputs_to_label(&outputs);

        println!("{}", prediction);
    }

    Ok(())
}

fn read_chess_file(path: &str) -> Result<Vec<String>, String> {
    let content =
        fs::read_to_string(path).map_err(|e| format!("Cannot read file {}: {}", path, e))?;

    let lines: Vec<String> = content
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .map(String::from)
        .collect();

    if lines.is_empty() {
        return Err(format!("File {} is empty", path));
    }

    Ok(lines)
}

fn extract_fen(line: &str) -> String {
    let parts: Vec<&str> = line.split_whitespace().collect();

    if parts.len() > 6 {
        parts[..6].join(" ")
    } else {
        line.to_string()
    }
}

fn outputs_to_label(outputs: &Vec<f64>) -> String {
    match outputs.len() {
        1 => {
            if outputs[0] > 0.5 {
                String::from("Check")
            } else {
                String::from("Nothing")
            }
        }
        2 => {
            if outputs[0] > outputs[1] {
                String::from("Nothing")
            } else {
                String::from("Check")
            }
        }
        3 => {
            let max_idx = find_max_index(outputs);
            match max_idx {
                0 => String::from("Nothing"),
                1 => String::from("Check"),
                2 => String::from("Checkmate"),
                _ => String::from("Unknown"),
            }
        }
        5 => {
            let max_idx = find_max_index(outputs);
            match max_idx {
                0 => String::from("Nothing"),
                1 => String::from("Check White"),
                2 => String::from("Check Black"),
                3 => String::from("Checkmate White"),
                4 => String::from("Checkmate Black"),
                _ => String::from("Unknown"),
            }
        }
        _ => {
            let max_idx = find_max_index(outputs);
            format!("Class {}", max_idx)
        }
    }
}

fn find_max_index(values: &Vec<f64>) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
