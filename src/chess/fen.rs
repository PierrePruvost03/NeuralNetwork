#[derive(Debug, Clone)]
pub struct FenPosition {
    pub board: [char; 64],
    pub active_color: char, // 'w' or 'b'
    pub castling: String,   // Droits de roque : combinaison de K, Q, k, q ou "-"
    pub en_passant: String, // Case en passant possible ou "-"
    pub halfmove: u32,      // Nombre de demi-coups depuis la dernière capture ou mouvement de pion
    pub fullmove: u32,      // Numéro du coup
}

impl FenPosition {
    // Format FEN : "pièces active_color castling en_passant halfmove fullmove"
    // Exemple : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    pub fn parse(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.trim().split_whitespace().collect();

        if parts.is_empty() {
            return Err(String::from("Empty FEN string"));
        }

        let board = Self::parse_board(parts[0])?;

        let active_color = if parts.len() > 1 {
            Self::parse_active_color(parts[1])?
        } else {
            'w'
        };

        let castling = if parts.len() > 2 {
            parts[2].to_string()
        } else {
            String::from("-")
        };

        let en_passant = if parts.len() > 3 {
            parts[3].to_string()
        } else {
            String::from("-")
        };

        let halfmove = if parts.len() > 4 {
            parts[4].parse().unwrap_or(0)
        } else {
            0
        };

        let fullmove = if parts.len() > 5 {
            parts[5].parse().unwrap_or(1)
        } else {
            1
        };

        Ok(FenPosition {
            board,
            active_color,
            castling,
            en_passant,
            halfmove,
            fullmove,
        })
    }

    fn parse_board(board_str: &str) -> Result<[char; 64], String> {
        let mut board = [' '; 64];
        let ranks: Vec<&str> = board_str.split('/').collect();

        if ranks.len() != 8 {
            return Err(format!(
                "Invalid FEN: expected 8 ranks, got {}",
                ranks.len()
            ));
        }

        for (rank_idx, rank) in ranks.iter().enumerate() {
            let mut file_idx = 0;

            for c in rank.chars() {
                if file_idx >= 8 {
                    return Err(format!(
                        "Invalid FEN: rank {} has too many squares",
                        rank_idx
                    ));
                }

                let board_idx = rank_idx * 8 + file_idx;

                if c.is_ascii_digit() {
                    let empty_count = c.to_digit(10).unwrap() as usize;
                    for i in 0..empty_count {
                        if file_idx + i >= 8 {
                            return Err(format!("Invalid FEN: rank {} overflow", rank_idx));
                        }
                        board[board_idx + i] = ' ';
                    }
                    file_idx += empty_count;
                } else if "prnbqkPRNBQK".contains(c) {
                    // Pièce valide
                    board[board_idx] = c;
                    file_idx += 1;
                } else {
                    return Err(format!("Invalid FEN: unknown character '{}'", c));
                }
            }

            if file_idx != 8 {
                return Err(format!(
                    "Invalid FEN: rank {} has {} squares instead of 8",
                    rank_idx, file_idx
                ));
            }
        }

        Ok(board)
    }

    fn parse_active_color(color: &str) -> Result<char, String> {
        match color {
            "w" => Ok('w'),
            "b" => Ok('b'),
            _ => Err(format!("Invalid active color: {}", color)),
        }
    }

    // Encode la position en vecteur d'inputs pour le réseau de neurones
    pub fn to_inputs(&self) -> Vec<f64> {
        let mut inputs = vec![0.0; 832];

        for (i, &piece) in self.board.iter().enumerate() {
            let piece_idx = Self::piece_to_index(piece);
            let input_idx = i * 13 + piece_idx;
            inputs[input_idx] = 1.0;
        }

        // TODO: On pourrait ajouter qui joue
        inputs
    }

    fn piece_to_index(piece: char) -> usize {
        match piece {
            ' ' => 0,
            'P' => 1,
            'N' => 2,
            'B' => 3,
            'R' => 4,
            'Q' => 5,
            'K' => 6,
            'p' => 7,
            'n' => 8,
            'b' => 9,
            'r' => 10,
            'q' => 11,
            'k' => 12,
            _ => 0, // Par défaut, case vide
        }
    }

    // Affiche l'échiquier de manière lisible (pour debug)
    #[allow(dead_code)]
    pub fn display(&self) {
        println!("  a b c d e f g h");
        for rank in 0..8 {
            print!("{} ", 8 - rank);
            for file in 0..8 {
                let idx = rank * 8 + file;
                let piece = self.board[idx];
                print!("{} ", if piece == ' ' { '.' } else { piece });
            }
            println!("{}", 8 - rank);
        }
        println!("  a b c d e f g h");
        println!("Active: {}", self.active_color);
        println!("Castling: {}", self.castling);
        println!("En passant: {}", self.en_passant);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_initial_position() {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let pos = FenPosition::parse(fen).unwrap();

        assert_eq!(pos.active_color, 'w');
        assert_eq!(pos.castling, "KQkq");
        assert_eq!(pos.board[0], 'r'); // a8
        assert_eq!(pos.board[4], 'k'); // e8
        assert_eq!(pos.board[63], 'R'); // h1
    }

    #[test]
    fn test_parse_empty_squares() {
        let fen = "8/8/8/8/8/8/8/8 w - - 0 1";
        let pos = FenPosition::parse(fen).unwrap();

        for piece in &pos.board {
            assert_eq!(*piece, ' ');
        }
    }

    #[test]
    fn test_to_inputs() {
        let fen = "8/8/8/8/8/8/8/K7 w - - 0 1";
        let pos = FenPosition::parse(fen).unwrap();
        let inputs = pos.to_inputs();

        assert_eq!(inputs.len(), 832); // 64 × 13

        // La case a1 (index 56) contient un roi blanc (K)
        // Donc l'input à l'index 56*13 + 6 devrait être 1.0
        assert_eq!(inputs[56 * 13 + 6], 1.0);
    }
}
