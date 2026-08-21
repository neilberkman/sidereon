use std::path::PathBuf;

use sidereon_core::data::{AnalysisCenter, NominalCoverageInterval, ProductDateTime, ProductType};
use sidereon_scoreboard::{
    default_lookback_days, parse_product_date, publication_status, report_json_pretty, run_default,
    utc_today, write_report_outputs, HttpsListingFetcher, PublicationStatusOutcome,
    ScoreboardError,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), ScoreboardError> {
    let cli = Cli::parse()?;
    if let Some((center, product_type)) = cli.publication_status {
        let at = cli.at.as_deref().ok_or_else(|| {
            ScoreboardError::InvalidArgument(
                "--publication-status requires --at YYYY-MM-DDTHH:MM:SSZ".to_string(),
            )
        })?;
        let now = parse_product_datetime(at)?;
        let outcome = publication_status(center, product_type, now, &HttpsListingFetcher)?;
        print!(
            "{}",
            render_publication_status(center, product_type, &outcome)
        );
        return Ok(());
    }
    if cli.at.is_some() {
        return Err(ScoreboardError::InvalidArgument(
            "--at requires --publication-status".to_string(),
        ));
    }
    let date = match cli.date {
        Some(value) => parse_product_date(&value)?,
        None => utc_today()?,
    };
    let report = run_default(date, cli.lookback_days)?;
    write_report_outputs(&report, cli.output.as_deref(), cli.history.as_deref())?;
    println!("{}", report_json_pretty(&report)?);
    Ok(())
}

#[derive(Debug)]
struct Cli {
    output: Option<PathBuf>,
    history: Option<PathBuf>,
    date: Option<String>,
    lookback_days: u32,
    publication_status: Option<(AnalysisCenter, ProductType)>,
    at: Option<String>,
}

impl Cli {
    fn parse() -> Result<Self, ScoreboardError> {
        let mut output = Some(PathBuf::from("latest.json"));
        let mut history = Some(PathBuf::from("history.jsonl"));
        let mut date = None;
        let mut lookback_days = default_lookback_days();
        let mut publication_status = None;
        let mut at = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--output" => output = Some(next_path(&mut args, "--output")?),
                "--history" => history = Some(next_path(&mut args, "--history")?),
                "--date" => date = Some(next_string(&mut args, "--date")?),
                "--lookback-days" => {
                    lookback_days = next_string(&mut args, "--lookback-days")?
                        .parse::<u32>()
                        .map_err(|_| {
                            ScoreboardError::InvalidArgument(
                                "--lookback-days must be an integer".to_string(),
                            )
                        })?;
                }
                "--publication-status" => {
                    let center = next_string(&mut args, "--publication-status CENTER")?
                        .parse::<AnalysisCenter>()?;
                    let product_type = next_string(&mut args, "--publication-status PRODUCT")?
                        .parse::<ProductType>()?;
                    publication_status = Some((center, product_type));
                }
                "--at" => at = Some(next_string(&mut args, "--at")?),
                "--stdout-only" => {
                    output = None;
                    history = None;
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    return Err(ScoreboardError::InvalidArgument(format!(
                        "unknown argument {other}"
                    )));
                }
            }
        }
        Ok(Self {
            output,
            history,
            date,
            lookback_days,
            publication_status,
            at,
        })
    }
}

fn next_path(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<PathBuf, ScoreboardError> {
    Ok(PathBuf::from(next_string(args, flag)?))
}

fn next_string(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, ScoreboardError> {
    args.next()
        .ok_or_else(|| ScoreboardError::InvalidArgument(format!("{flag} requires a value")))
}

fn print_help() {
    println!(
        "Usage: sidereon-scoreboard [--output latest.json] [--history history.jsonl] [--date YYYY-MM-DD] [--lookback-days N] [--stdout-only]\n       sidereon-scoreboard --publication-status CENTER PRODUCT --at YYYY-MM-DDTHH:MM:SSZ"
    );
}

fn parse_product_datetime(value: &str) -> Result<ProductDateTime, ScoreboardError> {
    let (date_text, time_text) = value
        .split_once('T')
        .ok_or_else(|| ScoreboardError::InvalidArgument("datetime must contain T".to_string()))?;
    let time_text = time_text
        .strip_suffix('Z')
        .ok_or_else(|| ScoreboardError::InvalidArgument("datetime must end in Z".to_string()))?;
    let mut parts = time_text.split(':');
    let hour = parse_time_part(parts.next(), "hour")?;
    let minute = parse_time_part(parts.next(), "minute")?;
    let second = parse_time_part(parts.next(), "second")?;
    if parts.next().is_some() {
        return Err(ScoreboardError::InvalidArgument(
            "datetime has extra time fields".to_string(),
        ));
    }
    ProductDateTime::new(parse_product_date(date_text)?, hour, minute, second)
        .map_err(ScoreboardError::from)
}

fn parse_time_part(value: Option<&str>, name: &str) -> Result<u8, ScoreboardError> {
    value
        .ok_or_else(|| ScoreboardError::InvalidArgument(format!("datetime missing {name}")))?
        .parse::<u8>()
        .map_err(|_| ScoreboardError::InvalidArgument(format!("datetime {name} is invalid")))
}

fn render_publication_status(
    center: AnalysisCenter,
    product_type: ProductType,
    outcome: &PublicationStatusOutcome,
) -> String {
    let mut output = format!("line: {center}/{product_type}\n");
    match outcome {
        PublicationStatusOutcome::Published {
            product,
            listing_url,
            behind_nominal_minutes,
            next_issue,
        } => {
            output.push_str(&format!(
                "published: {}\nlisting_url: {}\nlag_minutes: {}\n",
                product.filename, listing_url, behind_nominal_minutes,
            ));
            if let Some(next_issue) = next_issue {
                output.push_str(&format!(
                    "next_due: {}\nnext_identity: {}\n",
                    next_issue.due_at, next_issue.identity.official_filename,
                ));
                if let Some(interval) = next_issue.covers.observed {
                    output.push_str(&render_coverage("next_observed", interval));
                }
                if let Some(interval) = next_issue.covers.predicted {
                    output.push_str(&render_coverage("next_predicted", interval));
                }
            } else {
                output.push_str("next_due: unavailable (nominal schedule not cataloged)\n");
            }
        }
        PublicationStatusOutcome::NothingPublished { listing_urls } => {
            output.push_str(&format!(
                "status: nothing_published\nlisting_urls: {}\n",
                listing_urls.join(",")
            ));
        }
        PublicationStatusOutcome::Unreachable {
            listing_url,
            reason,
        } => output.push_str(&format!(
            "status: unreachable\nlisting_url: {listing_url}\nreason: {reason}\n"
        )),
    }
    output
}

fn render_coverage(label: &str, interval: NominalCoverageInterval) -> String {
    format!("{label}: {} through {}\n", interval.from, interval.until)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidereon_core::data::{next_issue_due, ProductDate, PublishedProduct};

    #[test]
    fn publication_status_output_places_next_due_beside_lag() {
        let date = ProductDate::new(2026, 8, 4).unwrap();
        let now = ProductDateTime::new(date, 7, 8, 0).unwrap();
        let next_issue = next_issue_due(AnalysisCenter::GfzUlt, ProductType::Sp3, now).unwrap();
        let outcome = PublicationStatusOutcome::Published {
            product: PublishedProduct {
                date: ProductDate::new(2026, 8, 3).unwrap(),
                issue: "0300".to_string(),
                filename: "GFZ0OPSULT_20262150300_02D_05M_ORB.SP3".to_string(),
                observed_at: Some("2026-08-04 08:20".to_string()),
            },
            listing_url: "https://example.invalid/".to_string(),
            behind_nominal_minutes: 1_688,
            next_issue: Some(Box::new(next_issue)),
        };
        let rendered =
            render_publication_status(AnalysisCenter::GfzUlt, ProductType::Sp3, &outcome);
        assert!(rendered.contains("lag_minutes: 1688"), "{rendered}");
        assert!(
            rendered.contains("next_due: 2026-08-04T08:50:00Z"),
            "{rendered}"
        );
        assert!(
            rendered.contains("next_identity: GFZ0OPSULT_20262150600_02D_05M_ORB.SP3"),
            "{rendered}"
        );
    }

    #[test]
    fn publication_status_datetime_parser_is_strict_utc() {
        assert_eq!(
            parse_product_datetime("2026-08-04T07:08:09Z").unwrap(),
            ProductDateTime::new(ProductDate::new(2026, 8, 4).unwrap(), 7, 8, 9).unwrap()
        );
        assert!(parse_product_datetime("2026-08-04 07:08:09Z").is_err());
        assert!(parse_product_datetime("2026-08-04T07:08:09").is_err());
    }
}
