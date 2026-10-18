// SPDX-License-Identifier: Apache-2.0
//! Reading back the log lines a test provoked. One global subscriber serves
//! every test, because a subscriber scoped to one test leaves the callsites
//! another test reaches first cached as disabled.

use std::cell::RefCell;
use std::io;
use std::sync::Once;

thread_local! {
    static CAPTURED: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Starts capturing for the current test thread, dropping what it had.
pub(crate) fn capture() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let subscriber = tracing_subscriber::fmt()
            .with_writer(ThreadCapture)
            .with_ansi(false)
            .finish();
        tracing::subscriber::set_global_default(subscriber).expect("no other global subscriber");
    });
    CAPTURED.with(|captured| captured.borrow_mut().clear());
}

/// Everything the current thread logged since `capture`.
pub(crate) fn captured() -> String {
    CAPTURED.with(|captured| String::from_utf8_lossy(&captured.borrow()).into_owned())
}

struct ThreadCapture;

impl io::Write for ThreadCapture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        CAPTURED.with(|captured| captured.borrow_mut().extend_from_slice(bytes));
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for ThreadCapture {
    type Writer = ThreadCapture;

    fn make_writer(&'a self) -> Self::Writer {
        ThreadCapture
    }
}
