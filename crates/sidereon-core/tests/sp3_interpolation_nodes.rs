//! Which nodes the precise-orbit position interpolator takes, and when it
//! refuses for want of them.
//!
//! RTKLIB `preceph.c` pephpos pivots on the last node strictly before the
//! query, takes eleven nodes starting five before the pivot, and refuses when
//! fewer than eleven are available. The fixture is synthetic: one GPS satellite
//! on a circular trajectory, a 300-second grid on 2020-06-25.

use sidereon_core::ephemeris::{EpochWindow, InterpolationNodes, Sp3};
use sidereon_core::Error;

/// A one-satellite product with `count` epochs from 00:00 at 300 s.
fn product(count: usize) -> Sp3 {
    let mut text = format!(
        "#cP2020  6 25  0  0  0.00000000     {count:3} ORBIT IGS14 FIT  TST\n\
         ## 2111 345600.00000000   300.00000000 59025 0.0000000000000\n\
         +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* SYNTHETIC SP3 NODE FIXTURE\n"
    );
    for index in 0..count {
        let angle = (index * 300) as f64 * core::f64::consts::TAU / 43_200.0;
        let x = 26_560.0 * libm::cos(angle);
        let y = 26_560.0 * libm::sin(angle) * 0.6;
        let z = 26_560.0 * libm::sin(angle) * 0.8;
        text.push_str(&format!(
            "*  2020  6 25 {:2}{:3}  0.00000000\nPG01{x:14.6}{y:14.6}{z:14.6}{:14.6}\n",
            index / 12,
            (index % 12) * 5,
            11.0
        ));
    }
    text.push_str("EOF\n");
    Sp3::parse(text.as_bytes()).expect("synthetic SP3")
}

/// Ten nodes are fewer than the eleven the interpolator takes: every query is
/// refused by name, including one on a node, where a shorter window would
/// otherwise return a lower-degree fit or the node itself. Eleven are served.
#[test]
fn a_run_shorter_than_eleven_nodes_is_refused_by_name() {
    let short = product(10);
    let sat = short.satellites()[0];
    let epochs = short.epochs_j2000_seconds();
    for query in [epochs[0], epochs[4] + 150.0, epochs[9]] {
        assert_eq!(
            short.position_at_j2000_seconds(sat, query),
            Err(Error::InsufficientPreciseNodes {
                sat,
                nodes: 10,
                required: 11,
            }),
            "{query}"
        );
    }

    let enough = product(11);
    let epochs = enough.epochs_j2000_seconds();
    assert!(enough
        .position_at_j2000_seconds(sat, epochs[5] + 150.0)
        .is_ok());
}

/// A query on node `k` pivots on node `k - 1` and takes nodes `k - 6` through
/// `k + 4`; just past the node it pivots on node `k` and takes `k - 5` through
/// `k + 5`. (At a node the interpolating polynomial passes through the node
/// whichever window is taken, so the pivot shows in the selected nodes and in
/// the rounding of the result, not in its value.)
#[test]
fn a_query_on_a_node_pivots_on_the_node_before_it() {
    let k = 20;
    let sp3 = product(40);
    let sat = sp3.satellites()[0];
    let epochs = sp3.epochs_j2000_seconds();
    let nodes = InterpolationNodes::for_sp3(&sp3);

    let on_node = EpochWindow::new(epochs[k], epochs[k]).expect("window");
    assert_eq!(
        nodes.selected_nodes(sat, on_node),
        epochs[k - 6..=k + 4].to_vec()
    );

    let past = epochs[k] + 1.0e-3;
    let past_node = EpochWindow::new(past, past).expect("window");
    assert_eq!(
        nodes.selected_nodes(sat, past_node),
        epochs[k - 5..=k + 5].to_vec()
    );

    // Before the first node the pivot is the first node.
    let first = EpochWindow::new(epochs[0], epochs[0]).expect("window");
    assert_eq!(nodes.selected_nodes(sat, first), epochs[..11].to_vec());
}
