use bitvec::{field::BitField, prelude::*};
use std::io::{BufRead, BufReader};

fn main() {
    // let graph = parse_input(std::io::stdin().lock());
    let graph = parse_input(BufReader::new(std::fs::File::open("graph2.in").unwrap()));
    // dbg!(graph);
    let cover = graph.compute_cover();
    dbg!(cover);
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
        let mut covered = vec![false; self.vertices as usize];
        for &vertex in &cover.vertecies {
            covered[vertex as usize - 1] = true;
            for &neighbour in self.neighbours(vertex) {
                covered[neighbour as usize - 1] = true;
            }
        }

        covered.iter().all(|&x| x)
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
            global_min: &mut u32,
            mut min: u32,
            ones: u32,
        ) -> (u32, Bits) {
            if vertices.is_empty() {
                return (ones, selected);
            }
            let n = vertices[0];
            let vertices = &vertices[1..];
            if ones > *global_min {
                return (*global_min + 1, selected);
            }
            let mut covered = selected[n as usize];
            let neighbours = graph.neighbours(n);
            for &neighbour in neighbours {
                covered |= selected[neighbour as usize];
            }
            if covered {
                return compute_cover_inner(graph, selected, vertices, global_min, min, ones);
            }
            let mut first = selected;
            first.set(n as usize, true);
            let (first_result, mut min_vec) =
                compute_cover_inner(graph, first, vertices, global_min, min, ones + 1);
            if first_result < *global_min {
                eprintln!("updating min from {} to {}", min, first_result);
                min = first_result;
                *global_min = min;
            }

            let mut second = selected;
            let neighbours = graph.neighbours(n);
            for &neighbour in graph.neighbours(n) {
                second.set(neighbour as usize, true);
            }
            let (result, second) = compute_cover_inner(
                graph,
                second,
                vertices,
                global_min,
                min,
                ones + neighbours.len() as u32,
            );
            if result < *global_min {
                eprintln!("updating min from {} to {}", min, result);
                min = result;
                min_vec = second;
                *global_min = min;
            }

            (min, min_vec)
        }

        let mut empty = bitarr![u64, Lsb0; 0; 301];

        // Actual reduction rules from the lecture
        for i in 1..=self.vertices {
            match self.neighbours(i) {
                [] => empty.set(i as usize, true),
                &[n] => {
                    eprintln!("hit rule 1");
                    empty.set(n as usize, true)
                }
                &[a, b] if self.neighbours(a).contains(&b) => {
                    eprintln!("hit rule 2.1");
                    empty.set(a as usize, true);
                    empty.set(b as usize, true);
                }
                &[a, b] if self.neighbours(a).len() == 2 && self.neighbours(b).len() == 2 => {
                    eprintln!("hit rule 2.3");
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
        let (min, vec) = compute_cover_inner(
            self,
            empty,
            &order,
            &mut (self.vertices / 2),
            u32::MAX,
            empty.count_ones() as u32,
        );
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
