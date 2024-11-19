use criterion::{criterion_group, criterion_main, Criterion};
use std::io::BufReader;
use vertex_cover::{parse_input, Graph};

fn benchmark_graph(c: &mut Criterion, graph_name: &str) {
    let graph = parse_input(BufReader::new(
        std::fs::File::open(format!("{}.in", graph_name)).unwrap(),
    ));

    c.bench_function(graph_name, |b| {
        b.iter(|| {
            let cover = graph.compute_cover();
            assert!(graph.validate_cover(&cover));
        })
    });
}

fn vertex_cover_benchmark(c: &mut Criterion) {
    // Benchmark graphs 0, 3, and 5
    benchmark_graph(c, "graph0");
    benchmark_graph(c, "graph3");
    benchmark_graph(c, "graph5");
}

criterion_group!(benches, vertex_cover_benchmark);
criterion_main!(benches);
