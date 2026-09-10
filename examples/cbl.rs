#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bincode::{DefaultOptions, Options};
use cbl::{kmer::Kmer, CBL};
use clap::{Args, Parser, Subcommand};
use const_format::formatcp;
use needletail::{parse_fastx_file, FastxReader};
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{stdout, BufReader, BufWriter, Write};
use std::path::Path;

// Loads runtime-provided constants for which declarations
// will be generated at `$OUT_DIR/constants.rs`.
pub mod constants {
    include!(concat!(env!("OUT_DIR"), "/constants.rs"));
}

use constants::{K, PREFIX_BITS, T};

// ============================================================================
// Allocateur instrumente (profilage memoire, portable macOS / Linux).
// Compte chaque allocation par classe de taille, sans jamais allouer lui-meme.
// Defini dans l'EXEMPLE : la bibliotheque cbl n'est pas modifiee.
// ============================================================================
mod allocstats {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

    const CLASSES: usize = 48;
    static COUNT: [AtomicUsize; CLASSES] = [const { AtomicUsize::new(0) }; CLASSES];
    static BYTES: [AtomicUsize; CLASSES] = [const { AtomicUsize::new(0) }; CLASSES];
    static NALLOC: AtomicUsize = AtomicUsize::new(0);
    static NREALLOC: AtomicUsize = AtomicUsize::new(0);
    static REQ: AtomicUsize = AtomicUsize::new(0);
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    #[inline]
    fn class_of(size: usize) -> usize {
        if size <= 1 {
            0
        } else {
            (usize::BITS - (size - 1).leading_zeros()) as usize
        }
    }
    #[inline]
    fn note(size: usize) {
        let c = class_of(size).min(CLASSES - 1);
        COUNT[c].fetch_add(1, Relaxed);
        BYTES[c].fetch_add(size, Relaxed);
        NALLOC.fetch_add(1, Relaxed);
        REQ.fetch_add(size, Relaxed);
        let live = LIVE.fetch_add(size, Relaxed) + size;
        PEAK.fetch_max(live, Relaxed);
    }
    #[inline]
    fn unnote(size: usize) {
        LIVE.fetch_sub(size, Relaxed);
    }

    pub struct Counting;
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            note(l.size());
            System.alloc(l)
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            note(l.size());
            System.alloc_zeroed(l)
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unnote(l.size());
            System.dealloc(p, l)
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            unnote(l.size());
            note(new);
            NREALLOC.fetch_add(1, Relaxed);
            System.realloc(p, l, new)
        }
    }

    /// Arrondi a la classe reelle de l'allocateur systeme.
    /// macOS : quanta de 16 o jusqu'a 1008 o, puis 512 o. glibc : 16 o, minimum 32 o.
    fn rounded(size: usize) -> usize {
        if cfg!(target_os = "macos") {
            if size == 0 {
                16
            } else if size <= 1008 {
                size.div_ceil(16) * 16
            } else if size <= 127 * 1024 {
                size.div_ceil(512) * 512
            } else {
                size.div_ceil(4096) * 4096
            }
        } else {
            let s = size + 8;
            if s <= 32 {
                32
            } else {
                s.div_ceil(16) * 16
            }
        }
    }

    pub fn report(label: &str) {
        let n = NALLOC.load(Relaxed);
        let req = REQ.load(Relaxed);
        eprintln!("\n===== PROFIL MEMOIRE : {label} =====");
        eprintln!("allocations             : {n}");
        eprintln!("reallocations           : {}", NREALLOC.load(Relaxed));
        eprintln!("octets demandes (cumul) : {:.3} Go", req as f64 / 1e9);
        eprintln!("pic vivant (demande)    : {:.3} Go", PEAK.load(Relaxed) as f64 / 1e9);
        eprintln!("vivant a la fin         : {:.3} Go", LIVE.load(Relaxed) as f64 / 1e9);
        eprintln!(
            "\n{:>10} {:>14} {:>12} {:>9} {:>13} {:>9}",
            "classe", "allocations", "demande Mo", "% alloc", "arrondi Mo", "surcout"
        );
        let mut tot_req = 0usize;
        let mut tot_round = 0usize;
        let mut small_round = 0usize;
        for c in 0..CLASSES {
            let k = COUNT[c].load(Relaxed);
            if k == 0 {
                continue;
            }
            let b = BYTES[c].load(Relaxed);
            let mid = if c == 0 { 1 } else { 1usize << c };
            let avg = (b / k).max(1);
            let r = k * rounded(avg);
            tot_req += b;
            tot_round += r;
            if mid <= 64 {
                small_round += r;
            }
            eprintln!(
                "{:>9}o {:>14} {:>11.1}M {:>8.2}% {:>12.1}M {:>8.2}x",
                mid,
                k,
                b as f64 / 1e6,
                100.0 * k as f64 / n.max(1) as f64,
                r as f64 / 1e6,
                r as f64 / b.max(1) as f64
            );
        }
        eprintln!(
            "{:>10} {:>14} {:>11.1}M {:>9} {:>12.1}M {:>8.2}x",
            "TOTAL",
            n,
            tot_req as f64 / 1e6,
            "",
            tot_round as f64 / 1e6,
            tot_round as f64 / tot_req.max(1) as f64
        );
        eprintln!(
            "\nclasses <= 64 o (allocations PAR SEAU) : {:.1} Mo arrondis, soit {:.1} % du total",
            small_round as f64 / 1e6,
            100.0 * small_round as f64 / tot_round.max(1) as f64
        );
    }
}

#[global_allocator]
static ALLOC: allocstats::Counting = allocstats::Counting;

#[derive(Parser, Debug)]
#[command(author, version, about = formatcp!("CBL compiled for K={K}"), long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build an index containing the k-mers of a FASTA/Q file
    Build(BuildArgs),
    /// Count the k-mers contained in an index
    Count(IndexArgs),
    /// Print a memory usage report of an index (instrumentation A1)
    Mem(IndexArgs),
    /// List the k-mers contained in an index
    List(ListArgs),
    /// Query an index for every k-mer contained in a FASTA/Q file
    Query(QueryArgs),
    /// Add the k-mers of a FASTA/Q file to an index
    Insert(UpdateArgs),
    /// Remove the k-mers of a FASTA/Q file from an index
    Remove(UpdateArgs),
    /// Compute the union of two indexes
    Merge(SetOpsArgs),
    /// Compute the intersection of two indexes
    Inter(SetOpsArgs),
    /// Compute the difference of two indexes
    Diff(SetOpsArgs),
    /// Compute the symmetric difference of two indexes
    SymDiff(SetOpsArgs),
    /// Show the repartition of the k-mers in the data structure
    Repartition(IndexArgs),
}

#[derive(Args, Debug)]
struct BuildArgs {
    /// Input file (FASTA/Q, possibly gzipped)
    input: String,
    /// Output file (no serialization by default)
    #[arg(short, long)]
    output: Option<String>,
    /// Use canonical k-mers
    #[arg(short, long)]
    canonical: bool,
}

#[derive(Args, Debug)]
struct IndexArgs {
    /// Index file (CBL format)
    index: String,
}

#[derive(Args, Debug)]
struct ListArgs {
    /// Index file (CBL format)
    index: String,
    /// Output file (write to stdout by default)
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(Args, Debug)]
struct QueryArgs {
    /// Index file (CBL format)
    index: String,
    /// Input file to query (FASTA/Q, possibly gzipped)
    input: String,
}

#[derive(Args, Debug)]
struct UpdateArgs {
    /// Index file (CBL format)
    index: String,
    /// Input file to query (FASTA/Q, possibly gzipped)
    input: String,
    /// Output file (no serialization by default)
    #[arg(short, long)]
    output: Option<String>,
}

#[derive(Args, Debug)]
struct SetOpsArgs {
    /// Index file (CBL format)
    first_index: String,
    /// Index file (CBL format)
    second_index: String,
    /// Output file (no serialization by default)
    #[arg(short, long)]
    output: Option<String>,
}

fn read_fasta<P: AsRef<Path> + Copy>(path: P) -> Box<dyn FastxReader> {
    parse_fastx_file(path)
        .unwrap_or_else(|_| panic!("Failed to open {}", path.as_ref().to_str().unwrap()))
}

fn read_index<D: DeserializeOwned, P: AsRef<Path> + Copy>(path: P) -> D {
    let index = File::open(path)
        .unwrap_or_else(|_| panic!("Failed to open {}", path.as_ref().to_str().unwrap()));
    let reader = BufReader::new(index);
    eprintln!(
        "Reading the index stored in {}",
        path.as_ref().to_str().unwrap()
    );
    DefaultOptions::new()
        .with_varint_encoding()
        .reject_trailing_bytes()
        .deserialize_from(reader)
        .unwrap()
}

fn write_index<S: Serialize, P: AsRef<Path> + Copy>(index: &S, path: P) {
    let output = File::create(path)
        .unwrap_or_else(|_| panic!("Failed to open {}", path.as_ref().to_str().unwrap()));
    let mut writer = BufWriter::new(output);
    eprintln!("Writing the index to {}", path.as_ref().to_str().unwrap());
    DefaultOptions::new()
        .with_varint_encoding()
        .reject_trailing_bytes()
        .serialize_into(&mut writer, &index)
        .unwrap();
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Command::Build(args) => {
            let input_filename = args.input.as_str();
            let mut cbl = if args.canonical {
                CBL::<K, T, PREFIX_BITS>::new_canonical()
            } else {
                CBL::<K, T, PREFIX_BITS>::new()
            };
            let mut reader = read_fasta(input_filename);
            if cbl.is_canonical() {
                eprintln!("Building the index of canonical {K}-mers contained in {input_filename}");
            } else {
                eprintln!("Building the index of {K}-mers contained in {input_filename}");
            }
            while let Some(record) = reader.next() {
                let seqrec = record.unwrap_or_else(|_| panic!("Invalid record"));
                cbl.insert_seq(&seqrec.seq());
            }
            {
                let (used, cap) = cbl.containers_load();
                eprintln!(
                    "conteneurs : {used} utilises / {cap} alloues ({:.1} % de remplissage)",
                    100.0 * used as f64 / cap.max(1) as f64
                );
            }
            allocstats::report("apres build, AVANT shrink_to_fit");
            cbl.shrink_to_fit();
            allocstats::report("apres build, APRES shrink_to_fit");
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::Count(args) => {
            let index_filename = args.index.as_str();
            let cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            if cbl.is_canonical() {
                eprintln!("It contains {} canonical {K}-mers", cbl.count());
            } else {
                eprintln!("It contains {} {K}-mers", cbl.count());
            }
        }
        Command::Mem(args) => {
            let index_filename = args.index.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            eprintln!("Memory report for {index_filename} (K={K})");
            print!("{}", cbl.memory_report());
            {
                let (used, cap) = cbl.containers_load();
                eprintln!(
                    "conteneurs : {used} utilises / {cap} alloues ({:.1} % de remplissage)",
                    100.0 * used as f64 / cap.max(1) as f64
                );
            }
            allocstats::report("index charge, AVANT shrink_to_fit");
            cbl.shrink_to_fit();
            {
                let (used, cap) = cbl.containers_load();
                eprintln!(
                    "conteneurs : {used} utilises / {cap} alloues ({:.1} % de remplissage)",
                    100.0 * used as f64 / cap.max(1) as f64
                );
            }
            allocstats::report("index charge, APRES shrink_to_fit");
        }
        Command::List(args) => {
            let index_filename = args.index.as_str();
            let cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            if cbl.is_canonical() {
                eprintln!("Listing canonical {K}-mers contained in {index_filename}");
            } else {
                eprintln!("Listing {K}-mers contained in {index_filename}");
            }
            if let Some(output_filename) = args.output {
                let output_filename = output_filename.as_str();
                let file = File::create(output_filename)
                    .unwrap_or_else(|_| panic!("Failed to open {}", output_filename));
                let mut writer = BufWriter::new(file);
                for kmer in cbl.iter() {
                    writer.write_all(&kmer.to_nucs()).unwrap();
                    writer.write_all(b"\n").unwrap();
                }
            } else {
                let mut writer = stdout().lock();
                for kmer in cbl.iter() {
                    writer.write_all(&kmer.to_nucs()).unwrap();
                    writer.write_all(b"\n").unwrap();
                }
            }
        }
        Command::Query(args) => {
            let index_filename = args.index.as_str();
            let input_filename = args.input.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            let mut reader = read_fasta(input_filename);
            if cbl.is_canonical() {
                eprintln!("Querying the canonical {K}-mers contained in {input_filename}");
            } else {
                eprintln!("Querying the {K}-mers contained in {input_filename}");
            }
            let mut total = 0usize;
            let mut positive = 0usize;
            while let Some(record) = reader.next() {
                let seqrec = record.expect("Invalid record");
                let contained = cbl.contains_seq(&seqrec.seq());
                total += contained.len();
                for p in contained {
                    if p {
                        positive += 1;
                    }
                }
            }
            eprintln!("# queries: {total}");
            eprintln!(
                "# positive queries: {positive} ({:.2}%)",
                (positive * 100) as f64 / total as f64
            );
        }
        Command::Insert(args) => {
            let index_filename = args.index.as_str();
            let input_filename = args.input.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            let mut reader = read_fasta(input_filename);
            if cbl.is_canonical() {
                eprintln!(
                    "Adding the canonical {K}-mers contained in {input_filename} to the index"
                );
            } else {
                eprintln!("Adding the {K}-mers contained in {input_filename} to the index");
            }
            while let Some(record) = reader.next() {
                let seqrec = record.expect("Invalid record");
                cbl.insert_seq(&seqrec.seq());
            }
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::Remove(args) => {
            let index_filename = args.index.as_str();
            let input_filename = args.input.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            let mut reader = read_fasta(input_filename);
            if cbl.is_canonical() {
                eprintln!(
                    "Removing the canonical {K}-mers contained in {input_filename} from the index"
                );
            } else {
                eprintln!("Removing the {K}-mers contained in {input_filename} from the index");
            }
            while let Some(record) = reader.next() {
                let seqrec = record.expect("Invalid record");
                cbl.remove_seq(&seqrec.seq());
            }
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::Merge(args) => {
            let first_index_filename = args.first_index.as_str();
            let second_index_filename = args.second_index.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(first_index_filename);
            let mut cbl2: CBL<K, T, PREFIX_BITS> = read_index(second_index_filename);
            cbl |= &mut cbl2;
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::Inter(args) => {
            let first_index_filename = args.first_index.as_str();
            let second_index_filename = args.second_index.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(first_index_filename);
            let mut cbl2: CBL<K, T, PREFIX_BITS> = read_index(second_index_filename);
            cbl &= &mut cbl2;
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::Diff(args) => {
            let first_index_filename = args.first_index.as_str();
            let second_index_filename = args.second_index.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(first_index_filename);
            let mut cbl2: CBL<K, T, PREFIX_BITS> = read_index(second_index_filename);
            cbl -= &mut cbl2;
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::SymDiff(args) => {
            let first_index_filename = args.first_index.as_str();
            let second_index_filename = args.second_index.as_str();
            let mut cbl: CBL<K, T, PREFIX_BITS> = read_index(first_index_filename);
            let mut cbl2: CBL<K, T, PREFIX_BITS> = read_index(second_index_filename);
            cbl ^= &mut cbl2;
            if let Some(output_filename) = args.output {
                write_index(&cbl, output_filename.as_str());
            }
        }
        Command::Repartition(args) => {
            let index_filename = args.index.as_str();
            let cbl: CBL<K, T, PREFIX_BITS> = read_index(index_filename);
            eprintln!(
                "{:.1}% of the available prefixes are used",
                cbl.prefix_load() * 100.0
            );
            let buckets_size_count = cbl.buckets_size_count();
            let total_buckets: usize = buckets_size_count.iter().map(|(_, &c)| c).sum();
            let total_items: usize = buckets_size_count.iter().map(|(&s, &c)| s * c).sum();
            eprintln!(
                "The average bucket size is {:.1} items",
                total_items as f64 / total_buckets as f64
            );
            let mut bucket_count = 0;
            let mut item_count = 0;
            for (&size, &count) in buckets_size_count.iter() {
                bucket_count += count;
                item_count += size * count;
                if count > total_buckets / 100 / 2
                    || size * count > total_items / 100 / 2
                    || bucket_count == total_buckets
                {
                    eprintln!(
                        "{:.1}% of items are in a bucket of size ≤ {size} ({:.1}% of buckets)",
                        (item_count * 100) as f64 / total_items as f64,
                        (bucket_count * 100) as f64 / total_buckets as f64,
                    );
                }
            }
            let (max_prefix, max_size) = cbl.buckets_sizes().max_by_key(|&(_, size)| size).unwrap();
            eprintln!("The biggest bucket (of size {max_size}) corresponds to prefix {max_prefix}");
            let buckets_node_count = cbl.buckets_node_count();
            let mut vec_count = 0;
            let mut vec_node_count = 0;
            let mut trie_count = 0;
            let mut trie_node_count = 0;
            for (&nodes, &count) in buckets_node_count.iter() {
                if nodes <= 1024 {
                    vec_count += count;
                    vec_node_count += nodes * count;
                } else {
                    trie_count += count;
                    trie_node_count += nodes * count;
                }
            }
            eprintln!(
                "{vec_count} vecs, average node count = {:.1}",
                vec_node_count as f64 / vec_count as f64
            );
            eprintln!(
                "{trie_count} tries, average node count = {:.1}",
                trie_node_count as f64 / trie_count as f64
            );
            let total_count = total_buckets + vec_node_count + trie_node_count;
            eprintln!("{total_count} nodes in total");
        }
    }
}
