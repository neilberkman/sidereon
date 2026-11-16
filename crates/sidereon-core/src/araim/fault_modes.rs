use std::cmp::Ordering;

use super::ism::Ism;
use super::{AraimError, AraimGeometry, IntegrityAllocation};
use crate::id::{GnssSatelliteId, GnssSystem};

/// One ARAIM fault hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultHypothesis {
    /// Satellites excluded by this hypothesis.
    pub excluded: Vec<GnssSatelliteId>,
    /// Constellation excluded by this hypothesis, if any.
    pub excluded_constellation: Option<GnssSystem>,
    /// Prior probability mass for this hypothesis.
    pub prior: f64,
}

impl FaultHypothesis {
    pub(crate) fn fault_free() -> Self {
        Self {
            excluded: Vec::new(),
            excluded_constellation: None,
            prior: 1.0,
        }
    }

    pub(crate) fn excludes_satellite(&self, id: GnssSatelliteId, system: GnssSystem) -> bool {
        self.excluded.contains(&id) || self.excluded_constellation == Some(system)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaultEnumeration {
    pub modes: Vec<FaultHypothesis>,
    pub p_unenumerated: f64,
}

/// Enumerate ARAIM primary-event combinations in descending probability order.
///
/// Mode probabilities follow ADD v3.0 Eq. (2): every selected primary event is
/// faulty and every unselected primary event is fault-free. Enumeration follows
/// ADD section 4.6 in descending probability within each fault order and stops
/// as soon as the remaining probability is below the unmonitored threshold.
/// Equal-probability modes retain lexicographic combination order over the ISM
/// event list (satellites in geometry order, then constellations), matching
/// MAAST/Octave's stable sort.
pub fn enumerate_fault_modes(
    geometry: &AraimGeometry,
    ism: &Ism,
    allocation: &IntegrityAllocation,
) -> Result<Vec<FaultHypothesis>, AraimError> {
    enumerate_fault_modes_checked(geometry, ism, allocation).map(|value| value.modes)
}

pub(crate) fn enumerate_fault_modes_checked(
    geometry: &AraimGeometry,
    ism: &Ism,
    allocation: &IntegrityAllocation,
) -> Result<FaultEnumeration, AraimError> {
    ism.validate()?;
    let mut events = Vec::with_capacity(geometry.rows.len() + ism.constellations.len());
    for row in &geometry.rows {
        events.push(PrimaryEvent::Satellite(
            row.id,
            ism.effective_for(row)?.p_sat,
        ));
    }
    let mut systems: Vec<_> = geometry.rows.iter().map(|row| row.system).collect();
    systems.sort_unstable();
    systems.dedup();
    for system in systems {
        events.push(PrimaryEvent::Constellation(
            system,
            ism.constellation(system)
                .ok_or(AraimError::InvalidIsm)?
                .p_const,
        ));
    }

    let p0: f64 = events.iter().map(|event| 1.0 - event.prior()).product();
    if !p0.is_finite() || p0 < 0.0 {
        return Err(AraimError::InvalidIsm);
    }
    let mut modes = vec![FaultHypothesis {
        prior: p0,
        ..FaultHypothesis::fault_free()
    }];
    let mut candidates = Vec::new();
    let max_order = allocation.max_fault_order.min(events.len());
    for order in 1..=max_order {
        collect_combinations(
            geometry,
            &events,
            order,
            0,
            &mut Vec::with_capacity(order),
            p0,
            &mut candidates,
        );
    }
    candidates.sort_by(compare_candidates);

    let order_masses = order_masses(&events);
    let mut p_unenumerated: f64 = order_masses.iter().skip(1).sum();
    for candidate in candidates {
        // ADD section 4.6 and MAAST `determine_subsets_v4` move modes one by
        // one, in this order, only while the remaining mass exceeds P_THRES.
        if p_unenumerated <= allocation.p_threshold_unmonitored {
            break;
        }
        p_unenumerated = (p_unenumerated - candidate.hypothesis.prior).max(0.0);
        modes.push(candidate.hypothesis);
    }
    // ADD v3.0 Eq. (16) applies FC to the sum of all same-constellation
    // satellite-fault modes, not to each mode separately. When the
    // inequality holds, Eq. (17) transfers that whole monitored probability
    // mass to the constellation-only representative.
    let constellation_modes: Vec<_> = modes
        .iter()
        .enumerate()
        .filter_map(|(index, mode)| {
            mode.excluded_constellation
                .filter(|_| mode.excluded.is_empty())
                .map(|system| (index, system, mode.prior))
        })
        .collect();
    let mut remove = vec![false; modes.len()];
    for (representative, system, constellation_prior) in constellation_modes {
        let candidates: Vec<_> = (1..modes.len())
            .filter(|&index| {
                index != representative
                    && !modes[index].excluded.is_empty()
                    // MAAST treats a composite constellation-plus-satellite
                    // event as belonging to C_j when every affected satellite
                    // is in j; use that reading where the ADD wording is open.
                    && modes[index]
                        .excluded_constellation
                        .is_none_or(|other| other == system)
                    && modes[index].excluded.iter().all(|id| id.system == system)
            })
            .collect();
        let consolidated_mass: f64 = candidates.iter().map(|&index| modes[index].prior).sum();
        if consolidated_mass <= 0.01 * constellation_prior {
            modes[representative].prior += consolidated_mass;
            for index in candidates {
                remove[index] = true;
            }
        }
    }
    modes = modes
        .into_iter()
        .enumerate()
        .filter_map(|(index, mode)| (!remove[index]).then_some(mode))
        .collect();
    Ok(FaultEnumeration {
        modes,
        p_unenumerated,
    })
}

#[derive(Debug, Clone, Copy)]
enum PrimaryEvent {
    Satellite(GnssSatelliteId, f64),
    Constellation(GnssSystem, f64),
}

impl PrimaryEvent {
    fn prior(self) -> f64 {
        match self {
            Self::Satellite(_, prior) | Self::Constellation(_, prior) => prior,
        }
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    hypothesis: FaultHypothesis,
    order: usize,
}

fn collect_combinations(
    geometry: &AraimGeometry,
    events: &[PrimaryEvent],
    order: usize,
    start: usize,
    selected: &mut Vec<usize>,
    p0: f64,
    out: &mut Vec<Candidate>,
) {
    if selected.len() == order {
        let mut excluded = Vec::new();
        let mut excluded_constellation = None;
        let mut prior = p0;
        for &idx in selected.iter() {
            let event = events[idx];
            let p = event.prior();
            if p == 0.0 {
                return;
            }
            prior *= p / (1.0 - p);
            match event {
                PrimaryEvent::Satellite(id, _) => excluded.push(id),
                PrimaryEvent::Constellation(system, _) if excluded_constellation.is_none() => {
                    excluded_constellation = Some(system);
                }
                PrimaryEvent::Constellation(system, _) => excluded.extend(
                    geometry
                        .rows
                        .iter()
                        .filter(|row| row.system == system)
                        .map(|row| row.id),
                ),
            }
        }
        excluded.sort_unstable();
        excluded.dedup();
        out.push(Candidate {
            hypothesis: FaultHypothesis {
                excluded,
                excluded_constellation,
                prior,
            },
            order,
        });
        return;
    }
    let remaining = order - selected.len();
    for idx in start..=events.len().saturating_sub(remaining) {
        selected.push(idx);
        collect_combinations(geometry, events, order, idx + 1, selected, p0, out);
        selected.pop();
    }
}

fn order_masses(events: &[PrimaryEvent]) -> Vec<f64> {
    let mut masses = vec![0.0; events.len() + 1];
    masses[0] = 1.0;
    for (seen, event) in events.iter().enumerate() {
        let p = event.prior();
        for order in (0..=seen + 1).rev() {
            masses[order] = masses[order] * (1.0 - p)
                + if order > 0 {
                    masses[order - 1] * p
                } else {
                    0.0
                };
        }
    }
    masses
}

fn compare_candidates(a: &Candidate, b: &Candidate) -> Ordering {
    // Rust's stable sort preserves the lexicographic combination-generation
    // order for equal probabilities, matching MAAST/Octave's stable `sort`.
    a.order.cmp(&b.order).then_with(|| {
        b.hypothesis
            .prior
            .partial_cmp(&a.hypothesis.prior)
            .unwrap_or(Ordering::Equal)
    })
}
