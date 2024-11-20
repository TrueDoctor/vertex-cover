// use bitvec::{field::BitField, prelude::*};
#![feature(stdarch_x86_avx512)]
use smallvec::{smallvec, SmallVec};
use std::{
    fmt::Write,
    io::{BufRead, BufReader},
};
use vertex_cover::*;

fn main() {
    // let graph = parse_input(std::io::stdin().lock());
    let graph = parse_input(BufReader::new(std::fs::File::open("graph2.in").unwrap()));
    // dbg!(graph);
    let cover = graph.compute_cover();
    // dbg!(cover);
    dbg!(graph.validate_cover(&cover));
    let translated = graph.translate_cover(&cover);
    std::fs::write("graph2.out", translated.format()).unwrap();
    println!("Hello, world!");
}
