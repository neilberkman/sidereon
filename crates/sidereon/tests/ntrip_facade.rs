use sidereon::ntrip::{
    format_gga, parse_sourcetable, ChunkedDecoder, GgaPosition, NtripClientMachine, NtripConfig,
    NtripCredentials, NtripEvent, NtripHandshake, NtripState, NtripVersion, StrAuth,
};

fn config(version: NtripVersion) -> NtripConfig {
    NtripConfig {
        host: "caster.example.test".into(),
        port: 2101,
        mountpoint: "MOUNT".into(),
        version,
        credentials: None,
        user_agent_product: "sidereon-test/0".into(),
        gga_interval_s: None,
    }
}

#[test]
fn facade_exposes_ntrip_request_stream_and_gga_behavior() {
    assert_eq!(
        config(NtripVersion::Rev1).request_bytes().expect("request"),
        b"GET /MOUNT HTTP/1.0\r\nUser-Agent: NTRIP sidereon-test/0\r\n\r\n"
    );

    let table = parse_sourcetable(
        "STR;MOUNT;ID;RTCM;1004;2;GPS;NET;USA;40.1;-105.2;1;0;gen;none;B;N;9600;misc\r\n\
         ENDSOURCETABLE\r\n",
    )
    .expect("sourcetable");
    let stream = table.streams().next().expect("stream record");
    assert_eq!(stream.mountpoint, "MOUNT");
    assert_eq!(stream.authentication, StrAuth::Basic);

    let mut decoder = ChunkedDecoder::new();
    assert_eq!(decoder.push(b"4\r\nWi").expect("chunk prefix"), b"Wi");
    assert_eq!(decoder.push(b"ki\r\n0\r\n\r\n").expect("chunk tail"), b"ki");
    assert!(decoder.finished());

    let mut machine = NtripClientMachine::new(config(NtripVersion::Rev2));
    assert_eq!(machine.state(), NtripState::Idle);
    machine.connection_request().expect("connection request");
    let events = machine.push(b"HTTP/1.1 200 OK\r\nContent-Type: gnss/data\r\n\r\nabc");
    assert_eq!(
        events,
        vec![
            NtripEvent::Connected(NtripHandshake {
                version: NtripVersion::Rev2,
                chunked: false,
                headers: vec![("Content-Type".into(), "gnss/data".into())],
            }),
            NtripEvent::Payload(b"abc".to_vec()),
        ]
    );
    assert_eq!(machine.state(), NtripState::Streaming);

    let sentence = format_gga(
        &GgaPosition {
            lat_deg: 40.0,
            lon_deg: -105.0,
            height_m: 1600.0,
            ..GgaPosition::default()
        },
        3661.239,
    )
    .expect("GGA sentence");
    assert_eq!(
        sentence,
        b"$GPGGA,010101.23,4000.0000000,N,10500.0000000,W,1,10,1.00,1600.0,M,,,,*2A\r\n"
    );
}

#[test]
fn facade_exposes_rev2_headers_and_chunked_machine() {
    let mut config = config(NtripVersion::Rev2);
    config.credentials = Some(NtripCredentials {
        username: "user".into(),
        password: "pass".into(),
    });
    assert_eq!(
        config.request_headers().expect("request headers"),
        (
            "/MOUNT".into(),
            vec![
                ("Host".into(), "caster.example.test:2101".into()),
                ("Ntrip-Version".into(), "Ntrip/2.0".into()),
                ("User-Agent".into(), "NTRIP sidereon-test/0".into()),
                ("Authorization".into(), "Basic dXNlcjpwYXNz".into()),
                ("Connection".into(), "close".into()),
            ],
        )
    );

    let mut machine = NtripClientMachine::new(config);
    machine.connection_request().expect("connection request");
    let events = machine.push(
        b"HTTP/1.1 200 OK\r\nContent-Type: gnss/data\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n0\r\n\r\n",
    );
    assert_eq!(
        events,
        vec![
            NtripEvent::Connected(NtripHandshake {
                version: NtripVersion::Rev2,
                chunked: true,
                headers: vec![
                    ("Content-Type".into(), "gnss/data".into()),
                    ("Transfer-Encoding".into(), "chunked".into()),
                ],
            }),
            NtripEvent::Payload(b"abc".to_vec()),
            NtripEvent::StreamEnded,
        ]
    );
    assert_eq!(machine.state(), NtripState::Closed);
}
