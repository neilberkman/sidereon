use std::collections::HashMap;
use std::sync::OnceLock;

use crate::astro::events::eclipse::fast_math::sin_cos;

use super::{earth_rotation_angle, ARCSEC_TO_RAD, CENTURY, DAY, TT_MINUS_TAI};

#[derive(Default)]
struct Term {
    multipliers: [i8; 15],
    coefficients: [Vec<[f64; 2]>; 3],
}

struct Series {
    terms: Vec<Term>,
    polynomials: [Vec<f64>; 3],
}

struct Conventions {
    arguments: Vec<Vec<f64>>,
    gmst: Vec<f64>,
    celestial: Series,
    tides: Series,
}

static CONVENTIONS: OnceLock<Conventions> = OnceLock::new();

pub(super) fn evaluate(tt: f64, tai: f64) -> ([f64; 3], [f64; 4]) {
    let tables = CONVENTIONS.get_or_init(Conventions::load);
    let tc = tt / CENTURY;
    let mut args = [0.0; 15];
    let mut rates = [0.0; 15];
    for (i, polynomial) in tables.arguments.iter().enumerate() {
        args[i + 1] = horner(polynomial, tc);
        rates[i + 1] = derivative(polynomial, tc);
    }
    let (celestial, _) = tables.celestial.evaluate(tc, &args, None);
    // The tidal series uses TT for both the Delaunay arguments and GMST.
    args[0] = (earth_rotation_angle(tai, TT_MINUS_TAI.as_seconds()) + horner(&tables.gmst, tc))
        + std::f64::consts::PI;
    rates[0] = super::ERA_1A + super::ERA_1B + derivative(&tables.gmst, tc);
    let (tide, ut1_rate) = tables.tides.evaluate(tc, &args, Some(&rates));
    (celestial, [tide[0], tide[1], tide[2], ut1_rate * -DAY])
}

impl Series {
    fn evaluate(&self, tc: f64, args: &[f64; 15], rates: Option<&[f64; 15]>) -> ([f64; 3], f64) {
        let mut high = [0.0; 3];
        let mut low = [0.0; 3];
        let mut ut1_rate = 0.0;
        for term in &self.terms {
            let (sin, cos) = sin_cos(term.argument(args));
            for (i, coefficients) in term.coefficients.iter().enumerate() {
                let [s, c] = coefficients
                    .iter()
                    .rev()
                    .fold([0.0; 2], |[s, c], &[a, b]| [s * tc + a, c * tc + b]);
                let value = s * sin + c * cos;
                let sum = high[i] + value;
                let high_part = sum - value;
                let value_part = sum - high_part;
                low[i] += (high[i] - high_part) + (value - value_part);
                high[i] = sum;
                // UT1 has only degree-zero terms in table 8.3.
                if let (2, Some(rates)) = (i, rates) {
                    let rate = term.argument(rates);
                    ut1_rate += (0.0 - c * rate) * sin + (0.0 + s * rate) * cos;
                }
            }
        }
        for (i, polynomial) in self.polynomials.iter().enumerate() {
            high[i] += low[i] + horner(polynomial, tc);
        }
        (high, ut1_rate)
    }
}

impl Term {
    fn argument(&self, arguments: &[f64; 15]) -> f64 {
        let m = &self.multipliers;
        let product = |i: usize| f64::from(m[i]) * arguments[i];
        let sum = |from, to| (from + 1..to).fold(product(from), |s, i| s + product(i));
        if m[0] != 0 {
            sum(0, 6)
        } else if m[1..6].iter().all(|&v| v == 0) {
            sum(6, 15)
        } else if m[6..].iter().all(|&v| v == 0) {
            sum(1, 6)
        } else if m[2] == 0 && m[12..].iter().all(|&v| v == 0) {
            // NoFarPlanetsTerm omits l' and the last three planetary arguments.
            (3..12).fold(product(1), |s, i| s + product(i))
        } else {
            sum(1, 15)
        }
    }

    fn add(&mut self, index: usize, degree: usize, pair: [f64; 2]) {
        if pair != [0.0; 2] {
            let coeff = &mut self.coefficients[index];
            if coeff.len() <= degree {
                coeff.resize(degree + 1, [0.0; 2]);
            }
            coeff[degree][0] += pair[0];
            coeff[degree][1] += pair[1];
        }
    }
}

const MICRO_ARCSEC: f64 = ARCSEC_TO_RAD * 1.0e-6;
const X: &str = include_str!("data/tab5.2a.txt");
const Y: &str = include_str!("data/tab5.2b.txt");
const S: &str = include_str!("data/tab5.2d.txt");
const GST: &str = include_str!("data/tab5.2e.txt");
const POLE: &str = include_str!("data/tab8.2ab.txt");
const UT1: &str = include_str!("data/tab8.3ab.txt");

impl Conventions {
    fn load() -> Self {
        let mut celestial = HashMap::new();
        let polynomials = [X, Y, S].map(|table| polynomial_part(table, radians(1.0 / 3.6e9)));
        for (index, table) in [X, Y, S].into_iter().enumerate() {
            let mut degree = 0;
            for line in table.lines() {
                let columns: Vec<_> = line.split_whitespace().collect();
                match columns.as_slice() {
                    ["j", "=", j, ..] => degree = integer(j),
                    [number, sin, cos, multipliers @ ..]
                        if multipliers.len() == 14 && number.parse::<usize>().is_ok() =>
                    {
                        let term = get_term(&mut celestial, 1, multipliers);
                        term.add(
                            index,
                            degree,
                            [MICRO_ARCSEC * real(sin), MICRO_ARCSEC * real(cos)],
                        );
                    }
                    _ => (),
                }
            }
        }
        let mut tides = HashMap::new();
        for (table, count) in [(POLE, 4), (UT1, 2)] {
            for line in table.lines() {
                let columns: Vec<_> = line.split_whitespace().collect();
                // A tide name is optional. Doodson number and period follow the multipliers.
                let required = 8 + count;
                if columns.len() < required {
                    continue;
                }
                let row = &columns[columns.len() - required..];
                if row[..6].iter().any(|column| column.parse::<i8>().is_err()) {
                    continue;
                }
                let term = get_term(&mut tides, 0, &row[..6]);
                if count == 4 {
                    term.add(
                        0,
                        0,
                        [MICRO_ARCSEC * real(row[8]), MICRO_ARCSEC * real(row[9])],
                    );
                    term.add(
                        1,
                        0,
                        [MICRO_ARCSEC * real(row[10]), MICRO_ARCSEC * real(row[11])],
                    );
                } else {
                    term.add(2, 0, [1.0e-6 * real(row[8]), 1.0e-6 * real(row[9])]);
                }
            }
        }
        Self {
            arguments: include_str!("data/nutation-arguments.txt")
                .lines()
                .filter_map(|line| line.split_once('='))
                .map(|(_, polynomial)| parse_polynomial(polynomial, 1.0))
                .collect(),
            gmst: polynomial_part(GST, radians(1.0 / 3600.0)),
            celestial: Series {
                terms: ordered(celestial, include_str!("data/celestial-order.txt")),
                polynomials,
            },
            tides: Series {
                terms: ordered(tides, include_str!("data/tidal-order.txt")),
                polynomials: Default::default(),
            },
        }
    }
}

fn get_term<'a>(
    terms: &'a mut HashMap<u64, Term>,
    start: usize,
    multipliers: &[&str],
) -> &'a mut Term {
    let mut all = [0_i8; 15];
    for (i, multiplier) in multipliers.iter().enumerate() {
        all[start + i] = multiplier.parse().expect("IERS multiplier");
    }
    let mut key = 0;
    let mut shift = 15;
    for (i, multiplier) in all.into_iter().enumerate() {
        if multiplier != 0 {
            key |= 1 << i;
            key |= ((i64::from(multiplier) + 64) as u64) << shift;
            shift += 7;
        }
    }
    terms.entry(key).or_insert_with(|| Term {
        multipliers: all,
        ..Term::default()
    })
}

fn ordered(mut terms: HashMap<u64, Term>, order: &str) -> Vec<Term> {
    let result = order
        .split_whitespace()
        .map(|key| {
            let key = u64::from_str_radix(key, 16).expect("IERS term key");
            terms.remove(&key).expect("IERS term coefficient")
        })
        .collect();
    assert!(terms.is_empty());
    result
}

fn real(value: &str) -> f64 {
    value.parse().expect("IERS coefficient")
}

fn integer(value: &str) -> usize {
    value.parse().expect("IERS degree")
}

fn polynomial_part(table: &str, unit: f64) -> Vec<f64> {
    let line = table
        .lines()
        .skip_while(|line| !line.starts_with("Polynomial part (unit"))
        .skip(1)
        .find(|line| !line.trim().is_empty())
        .expect("IERS polynomial");
    parse_polynomial(line, unit)
}

fn parse_polynomial(line: &str, default_unit: f64) -> Vec<f64> {
    let line = line.replace('−', "-").replace('×', "");
    let mut tokens = line.split_whitespace().peekable();
    let mut coefficients = Vec::new();
    let mut sign = 1.0;
    while let Some(token) = tokens.next() {
        match token {
            "+" => sign = 1.0,
            "-" => sign = -1.0,
            _ => {
                let end = token.find(['◦', '″', 't']).unwrap_or(token.len());
                let mut suffix = &token[end..];
                let unit = match suffix.chars().next() {
                    Some('◦') => {
                        suffix = suffix.trim_start_matches('◦');
                        radians(1.0)
                    }
                    Some('″') => {
                        suffix = suffix.trim_start_matches('″');
                        radians(1.0 / 3600.0)
                    }
                    _ => default_unit,
                };
                if suffix.is_empty() && tokens.peek().is_some_and(|t| t.starts_with('t')) {
                    suffix = tokens.next().expect("IERS power");
                }
                let degree = match suffix {
                    "" => 0,
                    "t" => 1,
                    "t²" => 2,
                    "t³" => 3,
                    "t⁴" => 4,
                    _ => integer(suffix.strip_prefix("t^").expect("IERS power")),
                };
                coefficients.resize(coefficients.len().max(degree + 1), 0.0);
                coefficients[degree] = sign * real(&token[..end]) * unit;
            }
        }
    }
    coefficients
}

// Hipparchus splits the conversion factor and the multiplicand.
fn radians(x: f64) -> f64 {
    let high = f64::from_bits(x.to_bits() & !((1 << 30) - 1));
    let low = x - high;
    low * 1.997844754509471e-9
        + low * 0.01745329052209854
        + high * 1.997844754509471e-9
        + high * 0.01745329052209854
}

fn horner(coefficients: &[f64], tc: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |v, &c| c + tc * v)
}

fn derivative(coefficients: &[f64], tc: f64) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .skip(1)
        .rev()
        .fold(0.0, |v, (i, c)| i as f64 * c + tc * v)
        / CENTURY
}
