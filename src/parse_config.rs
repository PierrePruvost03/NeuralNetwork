use std::env;

#[derive(Debug, PartialEq)]
pub enum Mode {
    Predict,
    Train,
}

#[derive(Debug)]
pub struct Config {
    pub mode: Mode,
    pub loadfile: String,
    pub chessfile: String,
    pub savefile: Option<String>,
    pub configfile: Option<String>,
}

impl Config {
    pub fn parse() -> Result<Self, String> {
        let args: Vec<String> = env::args().collect();

        if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
            Self::print_help();
            std::process::exit(0);
        }

        if args.len() < 4 {
            return Err(String::from("Not enough arguments"));
        }

        let mut mode: Option<Mode> = None;
        let mut savefile: Option<String> = None;
        let mut configfile: Option<String> = None;
        let mut loadfile: Option<String> = None;
        let mut chessfile: Option<String> = None;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--predict" => {
                    if mode.is_some() {
                        return Err(String::from("Cannot specify both --predict and --train"));
                    }
                    mode = Some(Mode::Predict);
                    i += 1;
                }
                "--train" => {
                    if mode.is_some() {
                        return Err(String::from("Cannot specify both --predict and --train"));
                    }
                    mode = Some(Mode::Train);
                    i += 1;
                }
                "--save" => {
                    if i + 1 >= args.len() {
                        return Err(String::from("--save requires a filename"));
                    }
                    i += 1;
                    savefile = Some(args[i].clone());
                    i += 1;
                }
                "--config" => {
                    if i + 1 >= args.len() {
                        return Err(String::from("--config requires a filename"));
                    }
                    i += 1;
                    configfile = Some(args[i].clone());
                    i += 1;
                }
                _ => {
                    if loadfile.is_none() {
                        loadfile = Some(args[i].clone());
                    } else if chessfile.is_none() {
                        chessfile = Some(args[i].clone());
                    } else {
                        return Err(format!("Unexpected argument: {}", args[i]));
                    }
                    i += 1;
                }
            }
        }

        let mode = mode.ok_or("Mode not specified (use --predict or --train)")?;
        let loadfile = loadfile.ok_or("LOADFILE not specified")?;
        let chessfile = chessfile.ok_or("CHESSFILE not specified")?;

        if savefile.is_some() && mode == Mode::Predict {
            return Err(String::from("--save can only be used with --train"));
        }

        Ok(Config {
            mode,
            loadfile,
            chessfile,
            savefile,
            configfile,
        })
    }

    pub fn print_help() {
        println!("USAGE");
        println!(
            "    ./my_torch_analyzer [--predict | --train [--save SAVEFILE] [--config CONFIGFILE]] LOADFILE CHESSFILE"
        );
        println!();
        println!("DESCRIPTION");
        println!("    --train       Launch the neural network in training mode. Each chessboard in FILE must");
        println!("                  contain inputs to send to the neural network in FEN notation and the expected output");
        println!("                  separated by space. If specified, the newly trained neural network will be saved in");
        println!(
            "                  SAVEFILE. Otherwise, it will be saved in the original LOADFILE."
        );
        println!();
        println!("    --predict     Launch the neural network in prediction mode. Each chessboard in FILE must");
        println!("                  contain inputs to send to the neural network in FEN notation, and optionally an expected");
        println!("                  output.");
        println!();
        println!("    --save        Save neural network into SAVEFILE. Only works in train mode.");
        println!();
        println!("    --config      Configuration file for training hyperparameters (.conf file).");
        println!("                  If not specified, uses default configuration.");
        println!("                  Only works in train mode.");
        println!();
        println!("    LOADFILE      File containing an artificial neural network");
        println!();
        println!("    CHESSFILE     File containing chessboards");
    }
}
