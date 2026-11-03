// SPDX-License-Identifier: Apache-2.0
//! Resident memory of the daemon and every process it started.

use std::collections::BTreeSet;

use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

/// Resident bytes of each process in a tree, one entry per process.
pub trait ProcessMemory {
    fn resident_bytes(&mut self) -> Vec<u64>;
}

/// Reads the tree rooted at the current process.
pub struct SystemMemory {
    system: System,
    root: Pid,
}

impl SystemMemory {
    pub fn new() -> Self {
        Self {
            system: System::new(),
            root: Pid::from_u32(std::process::id()),
        }
    }
}

impl ProcessMemory for SystemMemory {
    fn resident_bytes(&mut self) -> Vec<u64> {
        tree_members_rss(&mut self.system, self.root)
    }
}

/// The resident bytes of `root` and of every process descended from it.
fn tree_members_rss(system: &mut System, root: Pid) -> Vec<u64> {
    system.refresh_processes_specifics(
        ProcessesToUpdate::All,
        true,
        ProcessRefreshKind::nothing().with_memory(),
    );
    let mut members = BTreeSet::from([root]);
    loop {
        let before = members.len();
        // On Linux sysinfo lists each thread as a process that shares its
        // owner's memory, which would count that memory once per thread.
        for (pid, process) in system
            .processes()
            .iter()
            .filter(|(_, process)| process.thread_kind().is_none())
        {
            if process
                .parent()
                .is_some_and(|parent| members.contains(&parent))
            {
                members.insert(*pid);
            }
        }
        if members.len() == before {
            break;
        }
    }
    members
        .iter()
        .filter_map(|pid| system.process(*pid))
        .map(sysinfo::Process::memory)
        .collect()
}

/// Resident bytes of `root` plus every process descended from it.
pub(crate) fn tree_rss_bytes(system: &mut System, root: Pid) -> u64 {
    tree_members_rss(system, root).iter().sum()
}
