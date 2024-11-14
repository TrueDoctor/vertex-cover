use bitvec::{field::BitField, prelude::*};
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
    std::fs::write("cover", cover.format());
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
}

#[derive(Debug, Clone, Default)]
struct Cover {
    vertecies: Vec<u32>,
}

impl Cover {
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

    fn compute_cover(&self) -> Cover {
        fn compute_cover_inner(
            graph: &Graph,
            selected: Bits,
            vertices: &[u32],
            mut min: u32,
            ones: u32,
        ) -> (u32, Bits) {
            // eprintln!("{:?}", vertices);
            if vertices.is_empty() {
                #[cfg(debug_assertions)]
                let cover = Cover::from(selected);
                #[cfg(debug_assertions)]
                debug_assert!(graph.validate_cover(&cover));
                return (ones, selected);
            }
            let n = vertices[0];
            let vertices = &vertices[1..];
            if ones > min {
                // eprintln!("aborting with {} selected elements", ones);
                return (min + 1, selected);
            }
            let covered = selected[n as usize];
            let neighbours = graph.neighbours(n);
            let all = neighbours.iter().all(|n| selected[*n as usize]);
            let mut first = selected;
            if covered {
                // eprintln!("skipping {n} because it is already covered");
                return compute_cover_inner(graph, selected, vertices, min, ones);
            }
            let set = if !all {
                first.set(n as usize, true);
                1
            } else {
                0
            };
            // eprintln!("setting {}", n);
            // for neighbour in neighbours {
            //     eprintln!("n{}", neighbour);
            // }
            let (first_result, mut min_vec) =
                compute_cover_inner(graph, first, vertices, min, ones + set);
            if first_result < min {
                // eprintln!("updating min from {} to {}", min, first_result);
                min = first_result;
            }

            let mut second = selected;
            let neighbours = graph.neighbours(n);
            let mut neighbour_count = neighbours.len() as u32;
            for &neighbour in graph.neighbours(n) {
                // eprintln!("setting {}", neighbour);
                if second[neighbour as usize] {
                    neighbour_count -= 1;
                }
                second.set(neighbour as usize, true);
            }
            let (result, second) =
                compute_cover_inner(graph, second, vertices, min, ones + neighbour_count);
            if result < min {
                // eprintln!("updating min from {} to {}", min, result);
                min = result;
                min_vec = second;
            }

            (min, min_vec)
        }

        let mut empty = bitarr![u64, Lsb0; 0; 301];

        // Actual reduction rules from the lecture
        for i in 1..=self.vertices {
            match self.neighbours(i) {
                [] => empty.set(i as usize, true),
                &[n] => {
                    // eprintln!("hit rule 1");
                    empty.set(n as usize, true)
                }
                &[a, b] if self.neighbours(a).contains(&b) => {
                    // eprintln!("hit rule 2.1");
                    empty.set(a as usize, true);
                    empty.set(b as usize, true);
                }
                &[a, b] if self.neighbours(a).len() == 2 && self.neighbours(b).len() == 2 => {
                    // eprintln!("hit rule 2.3");
                    let intersection = self
                        .neighbours(a)
                        .iter()
                        .find(|n| self.neighbours(b).contains(n))
                        .unwrap();
                    empty.set(i as usize, true);
                    empty.set(*intersection as usize, true);
                }
                _ => (),
            }
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
