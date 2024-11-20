// use bitvec::{field::BitField, prelude::*};
#![feature(stdarch_x86_avx512)]
use bitvec::BitVec256;
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
    graph.reorder_ids();
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
    masks: Vec<[u64; 4]>,
    vertices: u32,
    id_map: Vec<u32>,
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
        self.neighbour_indices.clear();
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
        self.masks.clear();
        self.masks.push([0, 0, 0, 0]);
        self.neighbour_bits.clear();
        for i in 1..=self.vertices {
            let mut bits = Bits::default();
            for n in self.neighbours(i) {
                bits.set(*n as usize);
            }
            self.neighbour_bits.push(bits);
            let mut mask = Bits::new();
            mask.set(i as usize);
            self.masks.push(unsafe { mask.as_u64s() });
        }
    }

    fn reorder_ids(&mut self) {
        self.populate_neighbours();
        let mut order: Vec<u32> = (1..=self.vertices).collect();

        order.sort_by_key(|i| std::cmp::Reverse(self.neighbours(*i).len()));
        // order.sort_by_key(|i| self.neighbours(*i).len());
        let mut reverse_lookup = vec![0; order.len() + 1];
        for (i, id) in order.iter().enumerate() {
            reverse_lookup[*id as usize] = i + 1;
        }

        for (start, end) in self.edges.iter_mut() {
            *start = reverse_lookup[*start as usize] as u32;
            *end = reverse_lookup[*end as usize] as u32;
        }

        self.id_map = order;
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
                eprintln!("hit rule 2.1");
                smallvec![select(&[a, b], &[v])]
            }
            &[a, b]
                if self.deg(a) == 2 && self.deg(b) == 2 && self.union(a, b).count_ones() == 2 =>
            {
                eprintln!("hit rule 2.3");
                let intersection: Vec<u32> =
                    self.union(a, b).iter_ones().map(|x| x as u32).collect();
                smallvec![select(&intersection, &[a, b])]
            }
            &[a, b] if self.union(a, b).count_ones() >= 3 => {
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
    fn union(&self, a: u32, b: u32) -> Bits {
        self.neighbour_bits(a) | self.neighbour_bits(b)
    }

    pub fn compute_cover(&self) -> Cover {
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
        covered.set(0);

        // let mut best_min = self.vertices;

        // let mut best_cover = None;
        // for i in (self.vertices / 2)..(self.vertices) {
        //     // dbg!(i);
        //     let Some(cover) = self.compute_bounded_cover(selection, covered, i) else {
        //         continue;
        //     };
        //     best_cover = Some(cover);
        //     break;
        // }
        let Some(best_cover) = self.compute_bounded_cover(selection, covered, self.vertices - 1)
        else {
            panic!();
        };

        Cover::from(best_cover)
    }

    fn compute_bounded_cover(
        &self,
        selection: BitVec256,
        covered: BitVec256,
        mut best_min: u32,
    ) -> Option<BitVec256> {
        #[derive(Clone)]
        struct StackFrame {
            selected: Bits,
            finished: Bits,
            trailing_zeros_index: u8,
        }
        let mut stack = Vec::with_capacity(100);

        let mut best_cover = None;
        let mut frame = StackFrame {
            selected: selection,
            finished: covered,
            trailing_zeros_index: 0,
        };
        let mut i = 0u64;
        loop {
            i += 1;
            if i % 100000000 == 0 {
                // dbg!(stack.len());
            }
            let finished_bytes = unsafe { (!frame.finished).as_u64s() };
            while finished_bytes[frame.trailing_zeros_index as usize] == 0 {
                frame.trailing_zeros_index += 1;
            }
            let trailing_zeros = finished_bytes[frame.trailing_zeros_index as usize]
                .trailing_zeros()
                + frame.trailing_zeros_index as u32 * 64;
            let n = trailing_zeros;

            let ones = frame.selected.count_ones();
            if ones > best_min || n >= self.vertices {
                if ones < best_min {
                    // eprintln!("updating min from {} to {}", best_min, ones);
                    best_min = ones;
                    best_cover = Some(frame.selected);
                }
                let Some(new_frame) = stack.pop() else {
                    break;
                };
                frame = new_frame;
                continue;
            }

            if self.neighbour_bits(n).and_not(&frame.selected).is_zero() {
                frame.finished.set(n as usize);
                continue;
            }

            let branches = &self.branches[n as usize - 1];
            let first = branches[0];
            let branches = branches.iter().skip(1);
            for (new_bits, covered) in branches {
                let new_selection = *new_bits | frame.selected;
                let new_cover = *covered | frame.finished;

                stack.push(StackFrame {
                    selected: new_selection,
                    finished: new_cover,
                    trailing_zeros_index: frame.trailing_zeros_index,
                });
            }
            frame.selected |= first.0;
            frame.finished |= first.1;
        }
        best_cover
    }

    fn apply_reductions(&mut self) -> Vec<u32> {
        let mut vertex_cover = Vec::new();
        let mut modified = true;

        while modified {
            modified = false;

            // Apply degree-1 reduction
            for v in 1..=self.vertices {
                if let &[neighbour] = self.neighbours(v) {
                    // Take the single neighbor instead of v
                    vertex_cover.push(neighbour);
                    // Remove both vertices and their edges
                    self.edges
                        .retain(|&(a, b)| a != v && b != v && a != neighbour && b != neighbour);
                    self.populate_neighbours();
                    modified = true;
                    break;
                }
            }

            // Apply dominance rule
            if !modified {
                'outer: for v in 1..=self.vertices {
                    let v_neighbors = self.neighbour_bits(v);

                    // Check if v dominates any of its neighbors
                    for &u in self.neighbours(v) {
                        let u_neighbors = self.neighbour_bits(u);

                        // Check if N[u] ⊆ N[v] using BitVec operations
                        if u_neighbors.and_not(&v_neighbors).is_zero() {
                            vertex_cover.push(v);
                            self.edges.retain(|&(a, b)| a != v && b != v);
                            self.populate_neighbours();
                            modified = true;
                            break 'outer;
                        }
                    }
                }
            }
        }

        vertex_cover
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
