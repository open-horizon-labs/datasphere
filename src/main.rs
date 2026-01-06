use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "engram")]
#[command(about = "Background daemon that distills and links knowledge from local sources")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show daemon and graph statistics
    Stats,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Stats => {
            println!("engram stats (not yet implemented)");
        }
    }
}
