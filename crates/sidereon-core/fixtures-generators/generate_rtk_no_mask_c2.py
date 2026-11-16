#!/usr/bin/env python3
"""Generate the no-mask RTK regression fixture from commit c2a2aa6.

Run this script with a clean checkout of the recorded commit, outside the
working repository:

    python generate_rtk_no_mask_c2.py <historical-checkout> <output.json>

The temporary test uses the arc module's deterministic synthetic fixtures and
all five public solve/combination arc entry points and both RINEX builders.
State vectors use binary64 bit patterns; complete diagnostic records use
Rust's shortest round-trip decimals, preserving every finite binary64 value.
Residual reporting is enabled so the residual comparison is nonempty.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

COMMIT = "c2a2aa64f004e2238af5a609a7948f06a995037d"
MARKER = "RTK_NO_MASK_C2_FIXTURE="

TEST = r'''

    #[test]
    fn generate_rtk_no_mask_c2_fixture() {
        use serde_json::{json, Value};
        fn bits(values: &[f64]) -> Vec<u64> {
            values.iter().map(|value| value.to_bits()).collect()
        }
        fn arc(value: &RtkArcSolution) -> Value {
            json!({
                "references": value.references,
                "epoch_reported_baseline_bits": value.epochs.iter().map(|epoch| bits(&epoch.reported_baseline_m)).collect::<Vec<_>>(),
                "epoch_float_baseline_bits": value.epochs.iter().map(|epoch| bits(&epoch.float_baseline_m)).collect::<Vec<_>>(),
                "epoch_sd_ambiguity_bits": value.epochs.iter().map(|epoch| epoch.sd_ambiguities_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "epoch_used": value.epochs.iter().map(|epoch| &epoch.used_satellite_ids).collect::<Vec<_>>(),
                "epoch_fixed_ids": value.epochs.iter().map(|epoch| &epoch.fixed_ids).collect::<Vec<_>>(),
                "epoch_newly_fixed": value.epochs.iter().map(|epoch| &epoch.newly_fixed).collect::<Vec<_>>(),
                "epoch_fixed_dd_ids": value.epochs.iter().map(|epoch| &epoch.fixed_double_difference_ids).collect::<Vec<_>>(),
                "epoch_integer_fixed": value.epochs.iter().map(|epoch| epoch.integer_fixed).collect::<Vec<_>>(),
                "epoch_integer_ratio_bits": value.epochs.iter().map(|epoch| epoch.integer_ratio.to_bits()).collect::<Vec<_>>(),
                // Debug uses shortest round-trip binary64 decimal formatting.
                // Equality therefore preserves every represented finite bit.
                "epoch_residuals": value.epochs.iter().map(|epoch| format!("{:?}", epoch.residuals)).collect::<Vec<_>>(),
                "epoch_search": value.epochs.iter().map(|epoch| format!("{:?}", epoch.search)).collect::<Vec<_>>(),
                "epoch_geometry": value.epochs.iter().map(|epoch| format!("{:?}", epoch.geometry_quality)).collect::<Vec<_>>(),
                "state_baseline_bits": bits(&value.final_state.baseline_m),
                "state_sd_ambiguity_bits": bits(&value.final_state.sd_ambiguities_m),
                "state_information_bits": bits(&value.final_state.information),
                "measurement_covariance_bits": bits(&value.measurement_covariance),
                "fixed_cycles": value.final_state.fixed_cycles,
                "fixed_m_bits": value.final_state.fixed_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
            })
        }
        fn static_arc(value: &RtkStaticArcSolution) -> Value {
            json!({
                "references": value.references,
                "ambiguity_ids": value.ambiguity_ids,
                "float_diagnostics": format!("{:?}", value.float_solution),
                "fixed_diagnostics": format!("{:?}", value.fixed_solution),
                "geometry": format!("{:?}", value.geometry_quality),
                "float_baseline_bits": bits(&value.float_solution.baseline_m),
                "float_ambiguity_bits": value.float_solution.ambiguities_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
                "float_covariance_bits": bits(&value.float_solution.ambiguity_covariance_m),
                "fixed_baseline_bits": bits(&value.fixed_solution.fixed_solution.baseline_m),
                "fixed_cycles": value.fixed_solution.fixed_solution.fixed_ambiguities_cycles,
                "fixed_m_bits": value.fixed_solution.fixed_solution.fixed_ambiguities_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
            })
        }
        fn ionosphere_free(value: &RtkIonosphereFreeArcSolution) -> Value {
            json!({
                "references": value.references,
                "epochs": value.epochs.iter().map(|epoch| json!({
                    "base": epoch.base.iter().map(|obs| (&obs.satellite_id, &obs.ambiguity_id, obs.code_m.to_bits(), obs.phase_m.to_bits())).collect::<Vec<_>>(),
                    "rover": epoch.rover.iter().map(|obs| (&obs.satellite_id, &obs.ambiguity_id, obs.code_m.to_bits(), obs.phase_m.to_bits())).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                "wavelength_bits": value.wavelengths_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
                "offset_bits": value.offsets_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
            })
        }

        let (epochs, mut config, _, _) = integer_arc();
        config.update_opts.report_residuals = true;
        let sequential = solve_rtk_arc(&epochs, &config).expect("sequential");
        let static_config = RtkStaticArcConfig { arc: config.clone(), opts: validated_opts() };
        let static_solution = solve_static_rtk_arc(&epochs, &static_config).expect("static");

        let (dual_epochs, wide_config) = dual_arc();
        let wide = fix_wide_lane_rtk_arc(&dual_epochs, &wide_config).expect("wide lane");
        let if_config = RtkIonosphereFreeArcConfig {
            base_m: wide_config.base_m,
            initial_baseline_m: [0.0; 3],
            reference: wide_config.reference.clone(),
            apply_troposphere: false,
        };
        let ionosphere = prepare_ionosphere_free_rtk_arc(
            &wide.epochs, &wide.wide_lane_cycles, &if_config).expect("ionosphere free");
        let combined_static = solve_wide_lane_fixed_rtk_arc(
            &dual_epochs,
            &RtkWideLaneFixedArcConfig {
                wide_lane: wide_config.clone(),
                ionosphere_free: if_config.clone(),
                solve: RtkWideLaneFixedArcSolveConfig::Static(static_config),
            }).expect("combined static");
        let combined_sequential = solve_wide_lane_fixed_rtk_arc(
            &dual_epochs,
            &RtkWideLaneFixedArcConfig {
                wide_lane: wide_config,
                ionosphere_free: if_config,
                solve: RtkWideLaneFixedArcSolveConfig::Sequential(config),
            }).expect("combined sequential");
        let combined_static = match combined_static {
            RtkWideLaneFixedArcSolution::Static(value) => static_arc(&value.solution),
            _ => unreachable!(),
        };
        let combined_sequential = match combined_sequential {
            RtkWideLaneFixedArcSolution::Sequential(value) => arc(&value.solution),
            _ => unreachable!(),
        };
        let document = json!({
            "provenance": {"source_commit": "c2a2aa64f004e2238af5a609a7948f06a995037d", "generator": "fixtures-generators/generate_rtk_no_mask_c2.py"},
            "solve_rtk_arc": arc(&sequential),
            "solve_static_rtk_arc": static_arc(&static_solution),
            "fix_wide_lane_rtk_arc": {"references": wide.references, "wide_lane_cycles": wide.wide_lane_cycles, "diagnostics": format!("{wide:?}")},
            "prepare_ionosphere_free_rtk_arc": ionosphere_free(&ionosphere),
            "solve_wide_lane_fixed_rtk_arc_static": combined_static,
            "solve_wide_lane_fixed_rtk_arc_sequential": combined_sequential,
        });
        println!("RTK_NO_MASK_C2_FIXTURE={document}");
    }
'''


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != COMMIT:
        raise SystemExit(f"checkout is {actual}, expected {COMMIT}")
    source = checkout / "crates/sidereon-core/src/rtk_filter/arc.rs"
    original = source.read_text()
    anchor = "\n}\n\n#[cfg(test)]\n#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]\nmod carrier_arc_tests"
    if anchor not in original:
        raise SystemExit("arc test-module anchor not found")
    rinex_source = checkout / "crates/sidereon-core/src/rtk_filter/rinex_arc.rs"
    rinex_original = rinex_source.read_text()
    projection = Path(__file__).with_name("rtk_no_mask_rinex.rs").read_text()
    rinex_test = projection + '''
    #[test]
    fn generate_rtk_no_mask_c2_rinex_fixture() {
        println!("RTK_NO_MASK_C2_RINEX={}", historical_rinex_projection());
    }
'''
    end = rinex_original.rfind("}")
    rinex_source.write_text(rinex_original[:end] + rinex_test + rinex_original[end:])
    source.write_text(original.replace(anchor, TEST + anchor, 1))
    try:
        result = subprocess.run(
            ["cargo", "test", "-p", "sidereon-core", "--lib",
             "generate_rtk_no_mask_c2", "--", "--nocapture"],
            cwd=checkout, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True)
    finally:
        source.write_text(original)
        rinex_source.write_text(rinex_original)
    line = next(line for line in result.stdout.splitlines() if MARKER in line)
    document = json.loads(line.split(MARKER, 1)[1])
    rinex_line = next(line for line in result.stdout.splitlines() if "RTK_NO_MASK_C2_RINEX=" in line)
    document["rinex_arc_entry_points"] = json.loads(rinex_line.split("RTK_NO_MASK_C2_RINEX=", 1)[1])
    document["provenance"]["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    document["provenance"]["rinex_projection_sha256"] = hashlib.sha256(projection.encode()).hexdigest()
    document["provenance"]["diagnostic_encoding"] = "shortest round-trip binary64 decimals; exact comparison, including signed zero"
    document["provenance"]["report_residuals"] = True
    output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
