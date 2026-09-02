use std::fmt::Write as _;

use crate::format::fmtnum::fixed_decimals;

use super::sentence::checksum_body;
use super::{Gga, NmeaCoordinate, NmeaError, NmeaTalker, NmeaTime};

pub fn write_gga(talker: NmeaTalker, gga: &Gga) -> Result<String, NmeaError> {
    if let Some(time) = gga.time {
        if time.decimals != 2 {
            return Err(NmeaError::InvalidInput {
                field: "time",
                reason: "GGA writer requires NmeaTime.decimals == 2",
            });
        }
        if time.nanos % 10_000_000 != 0 {
            return Err(NmeaError::InvalidInput {
                field: "time",
                reason: "GGA writer emits exactly two fractional decimals",
            });
        }
    }
    if gga.latitude.is_some() != gga.longitude.is_some() {
        return Err(NmeaError::InvalidInput {
            field: "position",
            reason: "latitude and longitude must both be present or both absent",
        });
    }

    let talker = talker.code()?;
    let mut body = String::new();
    body.push(char::from(talker[0]));
    body.push(char::from(talker[1]));
    body.push_str("GGA,");
    push_opt(&mut body, gga.time.map(format_time));
    body.push(',');
    push_opt(
        &mut body,
        gga.latitude.map(|coord| format_coordinate(coord, true)),
    );
    body.push(',');
    if let Some(latitude) = gga.latitude {
        body.push(if latitude.negative { 'S' } else { 'N' });
    }
    body.push(',');
    push_opt(
        &mut body,
        gga.longitude.map(|coord| format_coordinate(coord, false)),
    );
    body.push(',');
    if let Some(longitude) = gga.longitude {
        body.push(if longitude.negative { 'W' } else { 'E' });
    }
    body.push(',');
    push_opt(
        &mut body,
        gga.quality.map(|quality| quality.value().to_string()),
    );
    body.push(',');
    push_opt(
        &mut body,
        gga.satellites_used
            .map(|satellites| format!("{satellites:02}")),
    );
    body.push(',');
    push_opt(&mut body, gga.hdop.map(|value| fixed_decimals(value, 2)));
    body.push(',');
    push_opt(
        &mut body,
        gga.altitude_msl_m.map(|value| fixed_decimals(value, 1)),
    );
    body.push(',');
    if gga.altitude_msl_m.is_some() {
        body.push('M');
    }
    body.push(',');
    push_opt(
        &mut body,
        gga.geoid_separation_m.map(|value| fixed_decimals(value, 1)),
    );
    body.push(',');
    if gga.geoid_separation_m.is_some() {
        body.push('M');
    }
    body.push(',');
    push_opt(
        &mut body,
        gga.differential_age_s.map(|value| fixed_decimals(value, 1)),
    );
    body.push(',');
    push_opt(
        &mut body,
        gga.differential_station_id
            .map(|station| format!("{station:04}")),
    );

    let checksum = checksum_body(&body);
    let mut sentence = String::with_capacity(body.len() + 6);
    let _ = write!(&mut sentence, "${body}*{checksum:02X}\r\n");
    Ok(sentence)
}

fn push_opt(body: &mut String, value: Option<String>) {
    if let Some(value) = value {
        body.push_str(&value);
    }
}

fn format_time(time: NmeaTime) -> String {
    format!(
        "{:02}{:02}{:02}.{:02}",
        time.hour,
        time.minute,
        time.second,
        time.nanos / 10_000_000
    )
}

fn format_coordinate(coordinate: NmeaCoordinate, is_latitude: bool) -> String {
    let degree_width = if is_latitude { 2 } else { 3 };
    let scale = 10_u64.pow(u32::from(coordinate.decimals));
    let whole_minutes = coordinate.minutes_scaled / scale;
    let fractional_minutes = coordinate.minutes_scaled % scale;
    if coordinate.decimals == 0 {
        format!("{:0degree_width$}{whole_minutes:02}", coordinate.degrees)
    } else {
        let frac_width = usize::from(coordinate.decimals);
        format!(
            "{:0degree_width$}{whole_minutes:02}.{fractional_minutes:0frac_width$}",
            coordinate.degrees
        )
    }
}
