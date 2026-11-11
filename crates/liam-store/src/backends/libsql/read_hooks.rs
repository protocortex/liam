// SPDX-License-Identifier: Apache-2.0
//! Observation points on the read path. In a normal build `ReadHooks` is a
//! zero-sized struct whose methods do nothing; a test build carries a
//! channel, a hold gate and barriers so a test can pause a statement and
//! watch how it ends.

#[cfg(not(test))]
use crate::error::Result;
#[cfg(not(test))]
use crate::value::Row;

#[cfg(not(test))]
#[derive(Clone)]
pub(super) struct ReadHooks;

#[cfg(not(test))]
impl ReadHooks {
    #[inline]
    pub(super) fn none() -> Self {
        Self
    }

    #[inline]
    pub(super) fn at_start(&self) {}

    #[inline]
    pub(super) fn before_step(&self) {}

    #[inline]
    pub(super) fn take_panic(&self) -> bool {
        false
    }

    #[inline]
    pub(super) fn count_interrupt(&self) {}

    #[inline]
    pub(super) fn skipped(&self) {}

    #[inline]
    pub(super) fn row_fetched(&self, _fetched: usize) -> Result<()> {
        Ok(())
    }

    #[inline]
    pub(super) fn statement_ended(&self) {}

    #[inline]
    pub(super) fn outcome(&self, _result: &Result<Vec<Row>>) {}
}

#[cfg(test)]
pub(super) use test_hooks::{ReadEvent, ReadHooks};

#[cfg(test)]
mod test_hooks {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc::{Receiver, Sender};
    use std::sync::{Arc, Barrier, Mutex};

    use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

    use crate::error::{Error, Result};
    use crate::value::Row;

    /// A held read that keeps going after its release stops here, so a
    /// statement nothing interrupts cannot grow without bound.
    const ROW_CAP: usize = 10_000;
    const ROW_CAP_MESSAGE: &str = "test row cap exceeded";

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(in crate::backends) enum ReadEvent {
        /// The first row is fetched: the statement is running in SQLite.
        Started,
        /// The blocking task is parked before its cancelled check.
        AtStart,
        /// The task is parked before a row step, after its cancelled check
        /// at the start.
        BeforeStep,
        /// The statement is over and the task is parked before its lease drops.
        StatementEnded,
        Ran,
        Interrupted,
        Skipped,
        /// The row cap stopped a statement nothing interrupted.
        RowCapped,
        Failed,
    }

    /// Cloned into each blocking read: the sender is per clone, everything
    /// else is shared with the backend's own copy.
    #[derive(Clone)]
    pub(in crate::backends) struct ReadHooks {
        events: Option<UnboundedSender<ReadEvent>>,
        shared: Arc<Shared>,
    }

    struct Shared {
        /// One shot: the first read to fetch a row takes it and waits on it.
        hold: Mutex<Option<Receiver<()>>>,
        held: AtomicBool,
        /// One shot, like `hold`: the first read to reach the point takes it.
        start_barrier: Mutex<Option<Arc<Barrier>>>,
        finish_barrier: Mutex<Option<Arc<Barrier>>>,
        step_barrier: Mutex<Option<Arc<Barrier>>>,
        panic_next: AtomicBool,
        interrupts: AtomicUsize,
    }

    impl ReadHooks {
        pub(in crate::backends) fn none() -> Self {
            Self {
                events: None,
                shared: Arc::new(Shared {
                    hold: Mutex::new(None),
                    held: AtomicBool::new(false),
                    start_barrier: Mutex::new(None),
                    finish_barrier: Mutex::new(None),
                    step_barrier: Mutex::new(None),
                    panic_next: AtomicBool::new(false),
                    interrupts: AtomicUsize::new(0),
                }),
            }
        }

        pub(in crate::backends) fn events(&mut self) -> UnboundedReceiver<ReadEvent> {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            self.events = Some(tx);
            rx
        }

        /// Arms the gate and returns the sender that releases it.
        pub(in crate::backends) fn arm_hold(&self) -> Sender<()> {
            let (tx, rx) = std::sync::mpsc::channel();
            *self.shared.hold.lock().unwrap_or_else(|e| e.into_inner()) = Some(rx);
            tx
        }

        /// A two party barrier: the read waits on it, the test is the other party.
        pub(in crate::backends) fn arm_start_barrier(&self) -> Arc<Barrier> {
            Self::arm(&self.shared.start_barrier)
        }

        pub(in crate::backends) fn arm_finish_barrier(&self) -> Arc<Barrier> {
            Self::arm(&self.shared.finish_barrier)
        }

        /// Parks the next read before its first row step. The row cap is armed
        /// too, so a read that is not stopped cannot run away.
        pub(in crate::backends) fn arm_step_barrier(&self) -> Arc<Barrier> {
            self.shared.held.store(true, Ordering::SeqCst);
            Self::arm(&self.shared.step_barrier)
        }

        fn arm(slot: &Mutex<Option<Arc<Barrier>>>) -> Arc<Barrier> {
            let barrier = Arc::new(Barrier::new(2));
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(barrier.clone());
            barrier
        }

        /// The next read panics once, then reads work again.
        pub(in crate::backends) fn arm_panic(&self) {
            self.shared.panic_next.store(true, Ordering::SeqCst);
        }

        pub(in crate::backends) fn interrupt_count(&self) -> usize {
            self.shared.interrupts.load(Ordering::SeqCst)
        }

        pub(in crate::backends) fn take_panic(&self) -> bool {
            self.shared.panic_next.swap(false, Ordering::SeqCst)
        }

        pub(in crate::backends) fn count_interrupt(&self) {
            self.shared.interrupts.fetch_add(1, Ordering::SeqCst);
        }

        pub(in crate::backends) fn at_start(&self) {
            let barrier = self
                .shared
                .start_barrier
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some(barrier) = barrier {
                self.send(ReadEvent::AtStart);
                barrier.wait();
            }
        }

        pub(in crate::backends) fn before_step(&self) {
            let barrier = self
                .shared
                .step_barrier
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some(barrier) = barrier {
                self.send(ReadEvent::BeforeStep);
                barrier.wait();
            }
        }

        pub(in crate::backends) fn skipped(&self) {
            self.send(ReadEvent::Skipped);
        }

        pub(in crate::backends) fn send(&self, event: ReadEvent) {
            if let Some(events) = &self.events {
                let _ = events.send(event);
            }
        }

        pub(in crate::backends) fn row_fetched(&self, fetched: usize) -> Result<()> {
            if fetched == 1 {
                let gate = self
                    .shared
                    .hold
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take();
                if let Some(gate) = gate {
                    self.shared.held.store(true, Ordering::SeqCst);
                    self.send(ReadEvent::Started);
                    // A dropped sender also releases the gate.
                    let _ = gate.recv();
                }
            }
            if self.shared.held.load(Ordering::SeqCst) && fetched > ROW_CAP {
                return Err(Error::Backend(ROW_CAP_MESSAGE.to_owned()));
            }
            Ok(())
        }

        pub(in crate::backends) fn statement_ended(&self) {
            let barrier = self
                .shared
                .finish_barrier
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
            if let Some(barrier) = barrier {
                self.send(ReadEvent::StatementEnded);
                barrier.wait();
            }
        }

        pub(in crate::backends) fn outcome(&self, result: &Result<Vec<Row>>) {
            let event = match result {
                Ok(_) => ReadEvent::Ran,
                Err(Error::Backend(message)) if message == ROW_CAP_MESSAGE => ReadEvent::RowCapped,
                Err(Error::Backend(message)) if message.contains("interrupt") => {
                    ReadEvent::Interrupted
                }
                Err(_) => ReadEvent::Failed,
            };
            self.send(event);
        }
    }
}
