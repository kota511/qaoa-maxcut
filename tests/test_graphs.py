import pytest

from qls_maxcut.graphs import load_gset_graph, make_weighted_erdos_renyi_graph


def test_weighted_erdos_renyi_graph_reproducible():
    graph_a = make_weighted_erdos_renyi_graph(10, 0.3, seed=123)
    graph_b = make_weighted_erdos_renyi_graph(10, 0.3, seed=123)

    weighted_edges_a = sorted(
        (u, v, data["weight"]) for u, v, data in graph_a.edges(data=True)
    )
    weighted_edges_b = sorted(
        (u, v, data["weight"]) for u, v, data in graph_b.edges(data=True)
    )

    assert weighted_edges_a == weighted_edges_b


def test_weighted_erdos_renyi_weights_are_positive_and_bounded():
    graph = make_weighted_erdos_renyi_graph(
        20,
        0.5,
        seed=123,
        weight_low=2,
        weight_high=5,
    )

    for _, _, data in graph.edges(data=True):
        assert 2 <= data["weight"] <= 5


def test_load_gset_graph_reads_a_one_based_matrix_market_file(tmp_path):
    path = tmp_path / "G1.mtx"
    path.write_text(
        "%%MatrixMarket matrix coordinate pattern symmetric\n"
        "% a comment the loader skips\n"
        "3 3 2\n1 2\n2 3\n"
    )

    graph = load_gset_graph(path)

    assert sorted(graph) == [0, 1, 2]
    assert graph[0][1]["weight"] == 1
    assert graph[1][2]["weight"] == 1
    assert len(graph.graph["sha256"]) == 64


def test_load_gset_graph_rejects_a_declared_edge_count_that_is_wrong(tmp_path):
    path = tmp_path / "G1.mtx"
    path.write_text("%%MatrixMarket matrix coordinate pattern symmetric\n3 3 5\n1 2\n")

    with pytest.raises(ValueError, match="edge count mismatch"):
        load_gset_graph(path)
