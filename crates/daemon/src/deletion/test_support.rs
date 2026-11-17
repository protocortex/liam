// SPDX-License-Identifier: Apache-2.0
//! A sealed store with two subjects, for the lineage store tests.

use std::path::Path;
use std::sync::Arc;

use liam_log::crypto::KeyRegistry;
use liam_store::{DefaultGraph, EventRole, LineageTarget, NewNode, NodeId};

use crate::config::{Config, EncryptionConfig, MasterKeySource};
use crate::log_startup::open_store_with_log;
use crate::log_startup::test_support::{config_in, database_in, embedder, log_dir_in};
use crate::shutdown::Shutdown;

pub const JOB: &str = "job-1";

pub struct Subjects {
    pub config: Config,
    pub keys: Arc<KeyRegistry>,
    pub store: Arc<DefaultGraph>,
    pub alice: NodeId,
    /// Alice's key group and the event that wrote her note, resolved.
    pub target: LineageTarget,
}

/// A store whose log is sealed, holding one note for alice and one for bob.
pub async fn sealed_subjects(dir: &Path) -> Subjects {
    sealed_subjects_with(dir, |_| {}).await
}

/// `sealed_subjects` over a config the caller has adjusted first.
pub async fn sealed_subjects_with(dir: &Path, adjust: impl FnOnce(&mut Config)) -> Subjects {
    let mut config = config_in(dir);
    adjust(&mut config);
    config.encryption = EncryptionConfig {
        master_key: Some(MasterKeySource::File),
        key_file: Some(dir.join("master.key").to_str().unwrap().to_owned()),
    };
    let started = open_store_with_log(
        &config,
        &database_in(dir),
        &log_dir_in(dir),
        embedder(),
        &Shutdown::default(),
    )
    .await
    .expect("start");
    let store = Arc::new(started.store);
    let mut ids = Vec::new();
    for subject in ["alice", "bob"] {
        let note =
            NewNode::now("fact", subject, format!("secret of {subject}")).with_subject(subject);
        ids.push(store.insert(note).await.expect("insert"));
    }
    let key = store
        .key_of_node(&ids[0])
        .await
        .unwrap()
        .expect("a key group");
    let events = store
        .node_event_ids(&ids[0], EventRole::Origin)
        .await
        .unwrap();
    let target = store
        .resolve_target(&LineageTarget {
            job_id: JOB.to_owned(),
            event_ids: events.into_iter().map(|event| event.event_id).collect(),
            key_ids: vec![key],
            ..LineageTarget::default()
        })
        .await
        .unwrap();
    Subjects {
        config,
        keys: started.keys.expect("a sealed log has a key registry"),
        store,
        alice: ids[0].clone(),
        target,
    }
}
