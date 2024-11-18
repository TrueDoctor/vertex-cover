use bitvec::{field::BitField, prelude::*};
use smallvec::{smallvec, SmallVec};
use std::{
    fmt::Write,
    io::{BufRead, BufReader},
};

fn main() {
    // let graph = parse_input(std::io::stdin().lock());
    let graph = parse_input(BufReader::new(std::fs::File::open("graph0.in").unwrap()));
    // dbg!(graph);
    let cover = graph.compute_cover();
    // dbg!(cover);
    dbg!(graph.validate_cover(&cover));
    std::fs::write("cover", cover.format()).unwrap();
    println!("Hello, world!");
}

fn parse_input(reader: impl BufRead) -> Graph {
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
struct Graph {
    edges: Vec<(u32, u32)>,
    neighbours: Vec<u32>,
    neighbour_indices: Vec<u32>,
    vertices: u32,
    branches: Vec<SmallVec<[Bits; 3]>>,
}

#[derive(Debug, Clone, Default)]
struct Cover {
    vertecies: Vec<u32>,
}

impl Cover {
    #[cfg(test)]
    fn full(n: u32) -> Cover {
        Cover {
            vertecies: (1..=n).collect(),
        }
    }
    fn format(&self) -> String {
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

type Bits = BitArray<[u64; 5]>;

impl Graph {
    fn validate_cover(&self, cover: &Cover) -> bool {
        for (start, end) in &self.edges {
            if !(cover.vertecies.contains(start) || cover.vertecies.contains(end)) {
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
    }

    fn neighbours(&self, vertex: u32) -> &[u32] {
        let start = self.neighbour_indices[vertex as usize - 1] as usize;
        let end = self.neighbour_indices[vertex as usize] as usize;
        &self.neighbours[start..end]
    }

    fn populate_branches(&mut self) {
        let mut branches = Vec::with_capacity(self.vertices as usize);
        for i in 1..=self.vertices {
            branches.push(self.compute_branches(i));
        }
        self.branches = branches;
    }
    fn compute_branches(&self, n: u32) -> SmallVec<[Bits; 3]> {
        let select = |values: &[u32]| {
            let mut values = values.to_vec();
            values.sort_unstable();
            values.dedup();
            let mut bits = Bits::default();
            for value in &values {
                bits.set(*value as usize, true);
            }
            bits
        };
        let neighbours = self.neighbours(n);

        match neighbours {
            [] => smallvec![select(&[n])],
            &[neighbour] => smallvec![select(&[neighbour])],
            &[a, b] if self.neighbours(a).contains(&b) => {
                // eprintln!("hit rule 2.1");
                smallvec![select(&[a, b])]
            }
            &[a, b] if self.neighbours(a).len() == 2 && self.neighbours(b).len() == 2 => {
                // eprintln!("hit rule 2.3");
                let intersection = self
                    .neighbours(a)
                    .iter()
                    .find(|n| self.neighbours(b).contains(n))
                    .unwrap();
                smallvec![select(&[n, *intersection])]
            }
            &[a, b] if self.neighbours(a).len() >= 2 && self.neighbours(b).len() >= 2 => {
                // eprintln!("hit rule 2.2");
                let mut neighbours = self.neighbours(a).to_vec();
                neighbours.extend_from_slice(self.neighbours(b));
                smallvec![select(&[a, b]), select(&neighbours)]
            }
            // &[a, b, c] if self.neighbours(a).contains(&b) => {
            //     // eprintln!("hit rule 2.1");
            //     smallvec![select(&[a, b])]
            // }
            _ => smallvec![select(&[n]), select(neighbours),],
        }
    }

    fn compute_cover(&self) -> Cover {
        fn compute_cover_inner(
            graph: &Graph,
            selected: Bits,
            vertices: &[u32],
            mut min: u32,
            ones: u32,
        ) -> (u32, Bits) {
            // dbg!(ones);
            assert_eq!(ones, selected.count_ones() as u32);
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
            if selected[n as usize] {
                // eprintln!("skipping {n} because it is already covered");
                return compute_cover_inner(graph, selected, vertices, min, ones);
            }

            let mut cover = selected;
            for new_bits in &graph.branches[n as usize - 1] {
                // eprintln!("new_bits for {n}: {:?}", new_bits);
                // dbg!(new_bits.count_ones());
                // dbg!(selected.count_ones());
                // eprintln!("current: {:?}", selected);
                let new_selection = *new_bits | selected;
                // eprintln!("combined: {:?}", new_selection);
                // dbg!(new_selection.count_ones());
                let (result, min_vec) = compute_cover_inner(
                    graph,
                    new_selection,
                    vertices,
                    min,
                    new_selection.count_ones() as u32,
                );
                if result < min {
                    eprintln!("updating min from {} to {}", min, result);
                    min = result;
                    cover = min_vec;
                }
            }

            (min, cover)
        }

        let mut empty = bitarr![u64, Lsb0; 0; 301];

        // Actual reduction rules from the lecture
        for i in 1..=self.vertices {
            let &[selection] = self.branches[i as usize - 1].as_slice() else {
                continue;
            };
            empty |= selection;
        }

        let mut order: Vec<u32> = (1..=self.vertices).collect();

        order.sort_by_key(|i| std::cmp::Reverse(self.neighbours(*i).len()));
        let max = self.vertices.min(self.edges.len() as u32);
        let (min, vec) = compute_cover_inner(self, empty, &order, max, empty.count_ones() as u32);
        dbg!(min);
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
        test_graph("graph0.in");
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

    fn test_graph(graph: &str) {
        let graph = parse_input(BufReader::new(std::fs::File::open(graph).unwrap()));
        let cover = graph.compute_cover();
        assert!(graph.validate_cover(&cover));
    }
}
