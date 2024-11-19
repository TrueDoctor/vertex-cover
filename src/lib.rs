// use bitvec::{field::BitField, prelude::*};
#![feature(stdarch_x86_avx512)]
use smallvec::{smallvec, SmallVec};
use std::{
    fmt::Write,
    io::{BufRead, BufReader},
};

pub mod bitvec;

pub fn parse_input(reader: impl BufRead) -> Graph {
    let mut lines = reader.lines();
    let sizes = lines
        .next()
        .expect("Empty input")
        .expect("unable to open stdin");
    let (vertices, edges) = sizes
        .split_once(' ')
        .expect("invalid format for first line");
    let vertex_count = vertices.parse().expect("invalid format for vertex count");
    let edge_count = edges.parse().expect("invalid format for edge count");

    let mut edges = Vec::with_capacity(edge_count);
    for line in lines {
        let edge = parse_edge(&line.unwrap());
        edges.push(edge);
    }
    let mut graph = Graph {
        edges,
        vertices: vertex_count,
        ..Default::default()
    };
    graph.populate_neighbours();
    graph.populate_branches();
    graph
}

fn parse_edge(line: &str) -> (u32, u32) {
    let (start, end) = line.split_once(' ').expect("invalid line format");
    let start = start.parse().unwrap();
    let end = end.parse().unwrap();
    (start, end)
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    edges: Vec<(u32, u32)>,
    neighbours: Vec<u32>,
    neighbour_bits: Vec<Bits>,
    neighbour_indices: Vec<u32>,
    vertices: u32,
    branches: Vec<SmallVec<[(Bits, Bits); 3]>>,
}

#[derive(Debug, Clone, Default)]
pub struct Cover {
    vertecies: Vec<u32>,
}

impl Cover {
    #[cfg(test)]
    fn full(n: u32) -> Cover {
        Cover {
            vertecies: (1..=n).collect(),
        }
    }
    pub fn format(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(&mut output, "{}", self.vertecies.len());
        for vertex in &self.vertecies {
            let _ = writeln!(&mut output, "{}", vertex);
        }
        output
    }
}

impl From<Bits> for Cover {
    fn from(value: Bits) -> Self {
        Cover {
            vertecies: value.iter_ones().map(|x| x as u32).collect(),
        }
    }
}

type Bits = bitvec::BitVec256;
// BitArray<[u64; 4]>;

impl Graph {
    pub fn validate_cover(&self, cover: &Cover) -> bool {
        for (start, end) in &self.edges {
            if !(cover.vertecies.contains(start) || cover.vertecies.contains(end)) {
                eprintln!("edge ({start}, {end}) not covered");
                return false;
            }
        }
        true
    }

    fn populate_neighbours(&mut self) {
        self.neighbours.clear();
        self.neighbour_indices.clear();
        fn neighbours<'a>(
            edges: &'a [(u32, u32)],
            vertex: &'a u32,
        ) -> impl Iterator<Item = u32> + 'a {
            edges
                .iter()
                .filter_map(|&(start, end)| match (start == *vertex, end == *vertex) {
                    (true, false) => Some(end),
                    (false, true) => Some(start),
                    _ => None,
                })
        }
        for i in 1..=self.vertices {
            self.neighbour_indices.push(self.neighbours.len() as u32);
            for neighbour in neighbours(&self.edges, &i) {
                self.neighbours.push(neighbour);
            }
        }
        self.neighbour_indices.push(self.neighbours.len() as u32);

        for (s, e) in &self.edges {
            if !self.neighbours(*s).contains(e) || !self.neighbours(*e).contains(s) {
                panic!();
            }
        }
        for i in 1..=self.vertices {
            let mut bits = Bits::default();
            for n in self.neighbours(i) {
                bits.set(*n as usize);
            }
            self.neighbour_bits.push(bits);
        }
    }

    fn neighbours(&self, vertex: u32) -> &[u32] {
        let start = self.neighbour_indices[vertex as usize - 1] as usize;
        let end = self.neighbour_indices[vertex as usize] as usize;
        &self.neighbours[start..end]
    }
    fn neighbour_bits(&self, vertex: u32) -> Bits {
        self.neighbour_bits[vertex as usize - 1]
    }

    fn populate_branches(&mut self) {
        let mut branches = Vec::with_capacity(self.vertices as usize);
        for i in 1..=self.vertices {
            branches.push(self.compute_branches(i));
        }
        self.branches = branches;
    }

    #[inline(never)]
    fn compute_branches(&self, v: u32) -> SmallVec<[(Bits, Bits); 3]> {
        let select = |values: &[u32], covered: &[u32]| {
            let mut values = values.to_vec();
            values.sort_unstable();
            values.dedup();
            let mut covered = covered.to_vec();
            covered.sort_unstable();
            covered.dedup();
            let mut bits = Bits::default();
            let mut processed = Bits::default();
            for value in &values {
                bits.set(*value as usize);
                processed.set(*value as usize);
            }
            for value in &covered {
                processed.set(*value as usize);
            }
            (bits, processed)
        };
        let neighbours = self.neighbours(v);

        let case3_2 = |a, b| self.connected(a, b);
        let select3_2 = |a, b, c| {
            eprintln!("hit rule 3.2");
            smallvec![select(&[a, b, c], &[v]), select(self.neighbours(c), &[c])]
        };
        let case3_3 = |a, b| self.intersect(a, b).count_ones() >= 2;
        let select3_3 = |a, b, c| {
            eprintln!("hit rule 3.3");
            let mut neighbours = self.intersect(a, b);
            neighbours.clear(v as usize);
            let common_neighbour = neighbours.iter_ones().next().unwrap();
            smallvec![
                select(&[v, common_neighbour as u32], &[]),
                select(&[a, b, c], &[v])
            ]
        };

        match neighbours {
            [] => smallvec![select(&[v], &[])],
            // This  would break with two conected vertecies of degree 1
            &[neighbour] => {
                eprintln!("hit rule 1");
                // eprintln!("selecting {neighbour}");
                smallvec![select(&[neighbour], &[v])]
            }
            &[a, b] if self.connected(a, b) => {
                // eprintln!("hit rule 2.1");
                smallvec![select(&[a, b], &[v])]
            }
            &[a, b]
                if self.deg(a) == 2
                    && self.deg(b) == 2
                    && self.intersect(a, b).count_ones() == 2 =>
            {
                eprintln!("hit rule 2.3");
                let intersection: Vec<u32> =
                    self.intersect(a, b).iter_ones().map(|x| x as u32).collect();
                smallvec![select(&intersection, &[a, b])]
            }
            &[a, b] if self.deg(a) >= 2 && self.deg(b) >= 2 => {
                eprintln!("hit rule 2.2");
                let mut neighbours = self.neighbours(a).to_vec();
                neighbours.extend_from_slice(self.neighbours(b));
                smallvec![select(&[a, b], &[v]), select(&neighbours, &[])]
            }
            &[a, b, c]
                if self.intersect(a, b).count_ones() == 1
                    && self.intersect(b, c).count_ones() == 1
                    && self.intersect(a, c).count_ones() == 1 =>
            {
                eprintln!("hit rule 3.1");
                let mut neighbours = vec![v, a];
                neighbours.extend_from_slice(self.neighbours(b));
                neighbours.extend_from_slice(self.neighbours(c));
                smallvec![
                    select(&neighbours, &[b, c]),
                    select(&[a, b, c], &[v]),
                    select(self.neighbours(a), &[a])
                ]
            }
            &[a, b, c] if case3_2(a, b) => select3_2(a, b, c),
            &[b, c, a] if case3_2(a, b) => select3_2(a, b, c),
            &[c, a, b] if case3_2(a, b) => select3_2(a, b, c),
            &[a, b, c] if case3_3(a, b) => select3_3(a, b, c),
            &[b, c, a] if case3_3(a, b) => select3_3(a, b, c),
            &[c, a, b] if case3_3(a, b) => select3_3(a, b, c),
            _ => smallvec![select(&[v], &[]), select(neighbours, &[v]),],
        }
    }

    fn connected(&self, a: u32, b: u32) -> bool {
        self.neighbour_bits(a)[b as usize]
    }

    fn deg(&self, a: u32) -> usize {
        self.neighbours(a).len()
    }

    fn intersect(&self, a: u32, b: u32) -> Bits {
        self.neighbour_bits(a) & self.neighbour_bits(b)
    }

    pub fn compute_cover(&self) -> Cover {
        fn compute_cover_inner(
            graph: &Graph,
            selected: Bits,
            mut finished: Bits,
            vertices: &[u32],
            mut min: u32,
            ones: u32,
        ) -> (u32, Bits) {
            // dbg!(ones);
            debug_assert_eq!(ones, selected.count_ones());
            // eprintln!("{:?}", selected);
            if vertices.is_empty() {
                #[cfg(debug_assertions)]
                let cover = Cover::from(selected);
                // dbg!(ones);
                #[cfg(debug_assertions)]
                debug_assert!(graph.validate_cover(&cover));
                return (ones, selected);
            }
            let n = vertices[0];
            // dbg!(ones);
            let vertices = &vertices[1..];
            if ones > min {
                // eprintln!("aborting with {} selected elements", ones);
                return (min + 1, selected);
            }
            if finished[n as usize] {
                // eprintln!("skipping {n} because it is already covered");
                return compute_cover_inner(graph, selected, finished, vertices, min, ones);
            }
            if graph.neighbour_bits(n).and_not(&selected).is_zero() {
                finished.set(n as usize);
                return compute_cover_inner(graph, selected, finished, vertices, min, ones);
            }

            let mut cover = selected;
            for (new_bits, covered) in &graph.branches[n as usize - 1] {
                // eprintln!("new_bits for {n}: {:?}", new_bits);
                // dbg!(new_bits.count_ones());
                // dbg!(selected.count_ones());
                // eprintln!("current: {:?}", selected);
                let new_selection = *new_bits | selected;
                let new_cover = *covered | finished;
                // eprintln!("combined: {:?}", new_selection);
                // dbg!(new_selection.count_ones());
                let (result, min_vec) = compute_cover_inner(
                    graph,
                    new_selection,
                    new_cover,
                    vertices,
                    min,
                    new_selection.count_ones(),
                );
                if result < min {
                    // eprintln!("updating min from {} to {}", min, result);
                    min = result;
                    cover = min_vec;
                }
            }

            (min, cover)
        }

        let mut selection: Bits = Default::default();
        let mut covered: Bits = Default::default();

        // Actual reduction when only one choice is possible
        for i in 1..=self.vertices {
            let &[(mut new_selection, new_covered)] = self.branches[i as usize - 1].as_slice()
            else {
                continue;
            };
            new_selection &= !covered;
            covered |= new_covered;
            selection |= new_selection;
        }

        let mut order: Vec<u32> = (1..=self.vertices).collect();
        // dbg!(selection.count_ones());
        // dbg!(covered.count_ones());
        // let covered_vec: Vec<usize> = covered.iter_ones().collect();
        // eprintln!("covered: {:?}", &covered_vec);

        order.sort_by_key(|i| std::cmp::Reverse(self.neighbours(*i).len()));
        let max = self.vertices.min(self.edges.len() as u32);
        let (min, vec) = compute_cover_inner(
            self,
            selection,
            covered,
            &order,
            max,
            selection.count_ones() as u32,
        );
        // dbg!(min);
        Cover::from(vec)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::io::BufReader;

    use crate::parse_input;

    #[test]
    fn test_full_cover() {
        let graph = parse_input(BufReader::new(std::fs::File::open("graph0.in").unwrap()));
        assert!(!graph.validate_cover(&Cover::default()));
        assert!(graph.validate_cover(&Cover::full(graph.vertices)));
    }

    #[test]
    fn test_graph0() {
        assert_eq!(test_graph("graph0.in"), 35);
    }
    #[test]
    fn test_graph1() {
        test_graph("graph1.in");
    }
    #[test]
    fn test_graph2() {
        test_graph("graph2.in");
    }
    #[test]
    fn test_graph3() {
        test_graph("graph3.in");
    }
    #[test]
    fn test_graph4() {
        test_graph("graph4.in");
    }
    #[test]
    fn test_graph5() {
        test_graph("graph5.in");
    }
    #[test]
    fn test_graph6() {
        test_graph("graph6.in");
    }

    fn test_graph(graph: &str) -> u32 {
        let graph = parse_input(BufReader::new(std::fs::File::open(graph).unwrap()));
        let cover = graph.compute_cover();
        assert!(graph.validate_cover(&cover));
        cover.vertecies.len() as u32
    }
}
