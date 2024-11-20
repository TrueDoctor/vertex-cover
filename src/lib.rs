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
    graph.init();
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
    component_mask: Bits,
    removed: Bits,
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
    fn init(&mut self) {
        self.populate_neighbours();
        self.reorder_ids();
        self.populate_neighbours();
        self.populate_branches();
        for i in 1..=self.vertices {
            self.component_mask.set(i as usize);
        }
    }

    fn populate_neighbours(&mut self) {
        self.neighbours.clear();
        self.neighbour_indices.clear();
        self.masks.clear();
        self.neighbour_bits.clear();
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
        self.masks.push([0, 0, 0, 0]);
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
    fn compute_branches(&mut self, v: u32) -> SmallVec<[(Bits, Bits); 3]> {
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
                // self.removed |= Bits::from_bit(v);
                // self.removed |= Bits::from_bit(neighbour);
                smallvec![select(&[neighbour], &[v])]
            }
            &[a, b] if self.connected(a, b) => {
                eprintln!("hit rule 2.1");
                // self.removed |= Bits::from_bit(a);
                // self.removed |= Bits::from_bit(b);
                // self.removed |= Bits::from_bit(v);
                smallvec![select(&[a, b], &[v])]
            }
            neigh
                if neigh.iter().any(|n| {
                    self.neighbour_bits(*n)
                        .and_not(&(self.neighbour_bits(v) | Bits::from_bit(v)))
                        .is_zero()
                        && !self.removed[*n as usize]
                }) =>
            {
                eprintln!("hit dominance rule");
                self.removed |= Bits::from_bit(v);
                smallvec![select(&[v], &[])]
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
        let components = self.find_connected_components();
        // let components = &[self];

        let mut selection: Bits = self.removed;
        let mut covered: Bits = self.removed;

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

        let mut final_cover = self.removed;
        // dbg!(components.len());
        for component in components {
            let covered = covered | !component.component_mask;
            // dbg!(covered.count_ones());
            let Some(best_cover) =
                component.compute_bounded_cover(selection, covered, component.vertices)
            else {
                dbg!(component.component_mask);
                panic!("did not find cover");
            };
            // dbg!(best_cover.count_ones());
            // dbg!(component.component_mask);
            final_cover |= best_cover;
        }

        // dbg!(final_cover.count_ones());
        // self.translate_cover(&Cover::from(final_cover))
        Cover::from(final_cover)
    }

    pub fn translate_cover(&self, cover: &Cover) -> Cover {
        let vertecies = cover
            .vertecies
            .iter()
            .map(|&v| self.id_map[v as usize - 1])
            .collect();
        Cover { vertecies }
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
        'outer: loop {
            i += 1;
            if i % 100000000 == 0 {
                dbg!(stack.len());
            }
            let finished_bytes = unsafe { (!frame.finished).as_u64s() };
            while finished_bytes[frame.trailing_zeros_index as usize] == 0 {
                frame.trailing_zeros_index += 1;
                if frame.trailing_zeros_index == 4 {
                    let ones = frame.selected.count_ones();
                    if ones < best_min {
                        // eprintln!("updating min from {} to {}", best_min, ones);
                        best_min = ones;
                        best_cover = Some(frame.selected);
                    }
                    let Some(new_frame) = stack.pop() else {
                        break 'outer;
                    };
                    frame = new_frame;
                    continue 'outer;
                }
            }
            let trailing_zeros = finished_bytes[frame.trailing_zeros_index as usize]
                .trailing_zeros()
                + frame.trailing_zeros_index as u32 * 64;
            let n = trailing_zeros;

            let ones = frame.selected.count_ones();
            if ones > best_min {
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

    fn find_connected_components(&self) -> Vec<Graph> {
        let mut visited = vec![false; self.vertices as usize + 1];
        let mut components = Vec::new();

        for v in 1..=self.vertices {
            if !visited[v as usize] {
                // Find all vertices in this component using DFS
                let mut component_vertices = Vec::new();
                let mut stack = vec![v];

                while let Some(current) = stack.pop() {
                    if !visited[current as usize] && !self.removed[current as usize] {
                        visited[current as usize] = true;
                        component_vertices.push(current);

                        // Add all unvisited neighbors to stack
                        for &neighbor in self.neighbours(current) {
                            if !visited[neighbor as usize] {
                                stack.push(neighbor);
                            }
                        }
                    }
                }

                if !component_vertices.is_empty() {
                    // Create subgraph for this component
                    let mut component_edges = Vec::new();
                    for &(start, end) in &self.edges {
                        if component_vertices.contains(&start) && component_vertices.contains(&end)
                        {
                            component_edges.push((start, end));
                        }
                    }
                    let mut component_mask = Bits::default();
                    for &vertex in &component_vertices {
                        component_mask.set(vertex as usize);
                    }

                    let mut neighbour_bits = self.neighbour_bits.clone();
                    neighbour_bits
                        .iter_mut()
                        .for_each(|neigh| *neigh &= component_mask);

                    let component = Graph {
                        edges: component_edges,
                        neighbour_bits,
                        component_mask,
                        ..self.clone()
                    };
                    // component.init();
                    components.push(component);
                }
            }
        }

        components
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
        dbg!(cover.vertecies.len() as u32);
        assert!(graph.validate_cover(&cover));
        cover.vertecies.len() as u32
    }
}
