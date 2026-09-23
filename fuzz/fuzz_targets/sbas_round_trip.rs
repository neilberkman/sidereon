#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::sbas::{SbasBlock, SbasWireForm};

const BODY_LEN: usize = 29;
const PREAMBLES: [u8; 3] = [0x53, 0x9A, 0xC6];

fn body_from_fuzz(data: &[u8]) -> [u8; BODY_LEN] {
    let mut body = [0u8; BODY_LEN];
    for (dst, src) in body.iter_mut().zip(data.iter()) {
        *dst = *src;
    }
    if let Some(&selector) = data.first() {
        body[0] = PREAMBLES[usize::from(selector) % PREAMBLES.len()];
    }
    if let Some(&message_type) = data.get(1) {
        body[1] = ((message_type & 0x3f) << 2) | (body[1] & 0x03);
    }
    body
}

fuzz_target!(|data: &[u8]| {
    let body = body_from_fuzz(data);
    let Ok(decoded) = SbasBlock::decode(&body, SbasWireForm::Body226) else {
        return;
    };

    let encoded = decoded.encode().expect("decoded SBAS block encodes");
    assert_eq!(encoded, body, "decoded SBAS block restates its bytes");
    let reparsed = SbasBlock::decode(&encoded, decoded.form).expect("encoded SBAS block decodes");
    assert_eq!(reparsed, decoded);
});
