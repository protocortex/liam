// SPDX-License-Identifier: Apache-2.0
//! The `[extraction]` section: turning log events into subject/predicate/object
//! facts. Off unless `enabled`.

use std::collections::{BTreeMap, HashMap};
use std::net::IpAddr;

use liam_store::{ArbitrationCaps, Cardinality, Grounding, MatchMode, PredicateSpec, Vocabulary};
use serde::Deserialize;

/// Trust given to a source the `[extraction.trust]` table does not name.
const DEFAULT_TRUST: f64 = 0.5;
/// Share of an LLM fact's source trust that it keeps: a model can misread.
const DEFAULT_LLM_TRUST_FACTOR: f64 = 0.8;
const DEFAULT_MIN_CONFIDENCE: f64 = 0.5;
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_CTX_TOKENS: usize = 4096;
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 120;
const DEFAULT_START_TIMEOUT_SECS: u64 = 120;
/// An hour is far past any model load; a larger value is a typo that would
/// also overflow the deadline arithmetic.
const MAX_START_TIMEOUT_SECS: u64 = 3600;
const DEFAULT_RUN_DEADLINE_SECS: u64 = 600;
/// A day is far past any extraction run; a larger value is a typo that would
/// overflow the deadline arithmetic.
pub const MAX_RUN_DEADLINE_SECS: u64 = 86_400;
const DEFAULT_RAM_BUDGET_MB: u64 = 4096;
const DEFAULT_MAX_EVENTS_PER_RUN: usize = 200;
const DEFAULT_BATCH_EVENTS: usize = 8;
const DEFAULT_MIN_EVENT_CHARS: usize = 40;
const DEFAULT_MAX_BATCH_FAILURES: u32 = 3;
/// Rough characters per token, to turn the context window into a text bound.
const CHARS_PER_TOKEN: usize = 4;
/// An event may take up to this share of the context window, the rest being
/// the prompt and the answer.
const EVENT_CONTEXT_SHARE_DIVISOR: usize = 2;
/// One batch of events may take up to this share of the context window, for
/// the same reason.
const BATCH_CONTEXT_SHARE_DIVISOR: usize = 2;

/// Server flags the guard sets itself, so `extra_args` may not repeat them.
/// `-a` is the short form of `--alias`.
const RESERVED_LONG_FLAGS: [&str; 4] = ["--host", "--port", "--api-key", "--alias"];
const RESERVED_SHORT_ALIAS: &str = "-a";

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExtractionConfig {
    /// When false the whole extraction hook is a no-op.
    pub enabled: bool,
    /// False runs the rules extractor only.
    pub llm_enabled: bool,
    pub llama_server_path: Option<String>,
    /// Appended to the server command line, for example thread limits.
    pub extra_args: Vec<String>,
    pub model_gguf: Option<String>,
    /// Loopback addresses only: the sidecar must not be reachable off-host.
    pub host: String,
    /// 0 picks a free port.
    pub port: u16,
    pub ctx_tokens: usize,
    pub request_timeout_secs: u64,
    pub start_timeout_secs: u64,
    pub run_deadline_secs: u64,
    pub ram_budget_mb: u64,
    pub max_events_per_run: usize,
    pub batch_events: usize,
    pub min_event_chars: usize,
    pub max_batch_failures: u32,
    /// Unset means derived from `ctx_tokens`, see `max_event_chars`.
    pub max_event_chars: Option<usize>,
    /// Phrases that end a fact, added to the built-in English and Portuguese.
    pub end_cues: Vec<String>,
    pub max_competing_per_key: usize,
    pub min_confidence: f64,
    pub llm_trust_factor: f64,
    pub default_trust: f64,
    /// Source name to trust score; `backfill` names the backfilled history.
    pub trust: HashMap<String, f64>,
    /// Predicate name to its rules, over the built-in vocabulary.
    pub predicates: HashMap<String, PredicateOverride>,
}

/// Fields left out keep the built-in value, or the undeclared-predicate
/// default for a predicate the built-ins do not know.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredicateOverride {
    pub cardinality: Option<Cardinality>,
    #[serde(rename = "match")]
    pub match_mode: Option<MatchMode>,
    pub grounding: Option<Grounding>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            llm_enabled: true,
            llama_server_path: None,
            extra_args: Vec::new(),
            model_gguf: None,
            host: DEFAULT_HOST.into(),
            port: 0,
            ctx_tokens: DEFAULT_CTX_TOKENS,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            start_timeout_secs: DEFAULT_START_TIMEOUT_SECS,
            run_deadline_secs: DEFAULT_RUN_DEADLINE_SECS,
            ram_budget_mb: DEFAULT_RAM_BUDGET_MB,
            max_events_per_run: DEFAULT_MAX_EVENTS_PER_RUN,
            batch_events: DEFAULT_BATCH_EVENTS,
            min_event_chars: DEFAULT_MIN_EVENT_CHARS,
            max_batch_failures: DEFAULT_MAX_BATCH_FAILURES,
            max_event_chars: None,
            end_cues: Vec::new(),
            max_competing_per_key: ArbitrationCaps::DEFAULT_MAX_COMPETING_PER_KEY,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            llm_trust_factor: DEFAULT_LLM_TRUST_FACTOR,
            default_trust: DEFAULT_TRUST,
            trust: HashMap::new(),
            predicates: HashMap::new(),
        }
    }
}

impl ExtractionConfig {
    /// Rejects values extraction cannot run with, naming the key to fix.
    pub fn validate(&self) -> anyhow::Result<()> {
        self.validate_host()?;
        self.validate_extra_args()?;
        self.validate_model_files()?;
        // A zero here makes extraction a silent no-op or an instant timeout.
        for (key, value) in [
            ("ctx_tokens", self.ctx_tokens as u64),
            ("request_timeout_secs", self.request_timeout_secs),
            ("start_timeout_secs", self.start_timeout_secs),
            ("run_deadline_secs", self.run_deadline_secs),
            ("max_events_per_run", self.max_events_per_run as u64),
            ("batch_events", self.batch_events as u64),
            ("max_competing_per_key", self.max_competing_per_key as u64),
            ("max_batch_failures", u64::from(self.max_batch_failures)),
        ] {
            anyhow::ensure!(value > 0, "extraction.{key} must be at least 1");
        }
        anyhow::ensure!(
            self.run_deadline_secs <= MAX_RUN_DEADLINE_SECS,
            "extraction.run_deadline_secs must be at most {MAX_RUN_DEADLINE_SECS}"
        );
        anyhow::ensure!(
            self.start_timeout_secs <= MAX_START_TIMEOUT_SECS,
            "extraction.start_timeout_secs must be at most {MAX_START_TIMEOUT_SECS}"
        );
        anyhow::ensure!(
            self.max_event_chars != Some(0),
            "extraction.max_event_chars must be at least 1"
        );
        for (key, value) in [
            ("min_confidence", self.min_confidence),
            ("llm_trust_factor", self.llm_trust_factor),
            ("default_trust", self.default_trust),
        ] {
            ensure_unit_interval(&format!("extraction.{key}"), value)?;
        }
        for (source, score) in &self.trust {
            ensure_unit_interval(&format!("extraction.trust.{source}"), *score)?;
        }
        Ok(())
    }

    /// The model pass cannot start without the model and the server that runs
    /// it, so an incomplete setup fails at load, not on every run.
    /// Expands `~` in the two file paths here, once, so every reader (the
    /// sidecar start, the stale sweep, the model hash, `liam fetch-models`)
    /// sees the same absolute path.
    pub(super) fn expand_paths(&mut self, home: &str) -> anyhow::Result<()> {
        for (key, value) in [
            ("extraction.llama_server_path", &mut self.llama_server_path),
            ("extraction.model_gguf", &mut self.model_gguf),
        ] {
            if let Some(text) = value.as_deref() {
                *value = Some(crate::models::resolve_path_with_home(key, text, home)?);
            }
        }
        Ok(())
    }

    fn validate_model_files(&self) -> anyhow::Result<()> {
        if !(self.enabled && self.llm_enabled) {
            return Ok(());
        }
        for (key, value) in [
            ("model_gguf", &self.model_gguf),
            ("llama_server_path", &self.llama_server_path),
        ] {
            anyhow::ensure!(
                value.is_some(),
                "extraction.{key} must be set while extraction.enabled and \
                 extraction.llm_enabled are on; set extraction.llm_enabled = false to run \
                 the rules alone"
            );
        }
        Ok(())
    }

    fn validate_host(&self) -> anyhow::Result<()> {
        let loopback = self
            .host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback());
        anyhow::ensure!(
            loopback,
            "extraction.host must be a loopback address such as {DEFAULT_HOST}, got {:?}",
            self.host
        );
        Ok(())
    }

    fn validate_extra_args(&self) -> anyhow::Result<()> {
        for arg in &self.extra_args {
            let reserved = RESERVED_LONG_FLAGS.iter().any(|flag| arg.starts_with(flag))
                || arg == RESERVED_SHORT_ALIAS
                || arg.starts_with(&format!("{RESERVED_SHORT_ALIAS}="));
            anyhow::ensure!(
                !reserved,
                "extraction.extra_args may not set {arg:?}: the daemon sets the address, \
                 port, key and alias of the sidecar itself"
            );
        }
        Ok(())
    }

    /// The most event text one batch holds: half the context window.
    pub fn max_batch_chars(&self) -> usize {
        self.ctx_tokens.saturating_mul(CHARS_PER_TOKEN) / BATCH_CONTEXT_SHARE_DIVISOR
    }

    /// The longest event text sent to the model: the configured bound, else
    /// half the context window.
    pub fn max_event_chars(&self) -> usize {
        self.max_event_chars.unwrap_or(
            self.ctx_tokens.saturating_mul(CHARS_PER_TOKEN) / EVENT_CONTEXT_SHARE_DIVISOR,
        )
    }
}

/// The built-in vocabulary with the configured predicates merged over it. The
/// one source of cardinality, match mode, and grounding for every extraction
/// step.
pub fn resolve_vocabulary(config: &ExtractionConfig) -> Vocabulary {
    Vocabulary::new(merged_specs(config))
}

/// Every predicate name the model may use, built-ins and configured, sorted.
pub fn declared_predicates(config: &ExtractionConfig) -> Vec<String> {
    merged_specs(config).into_keys().collect()
}

fn merged_specs(config: &ExtractionConfig) -> BTreeMap<String, PredicateSpec> {
    let mut specs: BTreeMap<String, PredicateSpec> = BUILT_IN_PREDICATES
        .iter()
        .map(|(name, spec)| ((*name).to_string(), *spec))
        .collect();
    for (name, over) in &config.predicates {
        let base = specs.get(name).copied().unwrap_or_default();
        specs.insert(
            name.clone(),
            PredicateSpec {
                cardinality: over.cardinality.unwrap_or(base.cardinality),
                match_mode: over.match_mode.unwrap_or(base.match_mode),
                grounding: over.grounding.unwrap_or(base.grounding),
            },
        );
    }
    specs
}

/// A number must be a real score in `[0.0, 1.0]`; NaN fails the range check.
fn ensure_unit_interval(key: &str, value: f64) -> anyhow::Result<()> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&value),
        "{key} must be between 0.0 and 1.0, got {value}"
    );
    Ok(())
}

const SINGLE_EXACT_TEXT: PredicateSpec = PredicateSpec {
    cardinality: Cardinality::One,
    match_mode: MatchMode::Exact,
    grounding: Grounding::Text,
};

/// Predicates the rules extractor and the model prompt know without any
/// configuration. A name left out is many-valued, which never destroys a fact.
const BUILT_IN_PREDICATES: [(&str, PredicateSpec); 11] = [
    (
        "lives_in",
        PredicateSpec {
            match_mode: MatchMode::Contains,
            ..SINGLE_EXACT_TEXT
        },
    ),
    ("works_at", SINGLE_EXACT_TEXT),
    ("has_title", SINGLE_EXACT_TEXT),
    (
        "born_on",
        PredicateSpec {
            grounding: Grounding::Date,
            ..SINGLE_EXACT_TEXT
        },
    ),
    ("status", SINGLE_EXACT_TEXT),
    ("knows", MANY_EXACT_TEXT),
    ("member_of", MANY_EXACT_TEXT),
    ("mentioned_in", MANY_EXACT_TEXT),
    ("references_date", MANY_EXACT_TEXT),
    ("references_url", MANY_EXACT_TEXT),
    ("references_id", MANY_EXACT_TEXT),
];

const MANY_EXACT_TEXT: PredicateSpec = PredicateSpec {
    cardinality: Cardinality::Many,
    match_mode: MatchMode::Exact,
    grounding: Grounding::Text,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use tempfile::TempDir;

    fn load(toml: &str) -> anyhow::Result<Config> {
        let dir = TempDir::new().expect("create temp dir");
        let path = dir.path().join("liam.toml");
        std::fs::write(&path, toml).expect("write config");
        Config::load(&path)
    }

    fn loaded_extraction(toml: &str) -> ExtractionConfig {
        load(toml).expect("config loads").extraction
    }

    fn error_of(toml: &str) -> String {
        format!("{:#}", load(toml).expect_err("config must be rejected"))
    }

    #[test]
    fn a_start_timeout_over_the_cap_is_rejected_and_the_cap_itself_is_allowed() {
        // Arrange
        let over = format!(
            "[extraction]\nstart_timeout_secs = {}\n",
            MAX_START_TIMEOUT_SECS + 1
        );
        let at = format!("[extraction]\nstart_timeout_secs = {MAX_START_TIMEOUT_SECS}\n");

        // Act
        let message = error_of(&over);
        let allowed = loaded_extraction(&at);

        // Assert
        assert!(message.contains("start_timeout_secs"), "{message}");
        assert_eq!(allowed.start_timeout_secs, MAX_START_TIMEOUT_SECS);
    }

    #[test]
    fn the_model_pass_needs_the_model_and_its_server() {
        // Arrange: (toml, whether the load must succeed)
        let files = "model_gguf = \"m.gguf\"\nllama_server_path = \"/bin/llama-server\"\n";
        let rows = [
            ("enabled = true\n".to_owned(), false),
            (
                "enabled = true\nmodel_gguf = \"m.gguf\"\n".to_owned(),
                false,
            ),
            (
                "enabled = true\nllama_server_path = \"/bin/llama-server\"\n".to_owned(),
                false,
            ),
            (format!("enabled = true\n{files}"), true),
            ("enabled = true\nllm_enabled = false\n".to_owned(), true),
            ("enabled = false\n".to_owned(), true),
        ];

        for (body, accepted) in rows {
            // Act
            let result = load(&format!("[extraction]\n{body}"));

            // Assert
            assert_eq!(result.is_ok(), accepted, "{body:?}: {result:?}");
        }
    }

    #[test]
    fn the_missing_file_setting_is_named() {
        // Arrange / Act
        let message = error_of("[extraction]\nenabled = true\n");

        // Assert
        assert!(message.contains("extraction.model_gguf"), "{message}");
    }

    #[test]
    fn an_unknown_key_under_extraction_fails_loudly() {
        // Arrange
        let toml = "[extraction]\nbogus = 1\n";

        // Act
        let message = error_of(toml);

        // Assert
        assert!(message.contains("bogus"), "{message}");
    }

    #[test]
    fn no_extraction_section_gives_the_documented_defaults() {
        // Arrange / Act
        let config = loaded_extraction("");

        // Assert
        assert_eq!(config, ExtractionConfig::default());
        assert!(!config.enabled);
        assert!(config.llm_enabled);
        assert_eq!(config.llama_server_path, None);
        assert!(config.extra_args.is_empty());
        assert_eq!(config.model_gguf, None);
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 0);
        assert_eq!(config.ctx_tokens, 4096);
        assert_eq!(config.request_timeout_secs, 120);
        assert_eq!(config.start_timeout_secs, 120);
        assert_eq!(config.run_deadline_secs, 600);
        assert_eq!(config.ram_budget_mb, 4096);
        assert_eq!(config.max_events_per_run, 200);
        assert_eq!(config.batch_events, 8);
        assert_eq!(config.min_event_chars, 40);
        assert_eq!(config.max_batch_failures, 3);
        assert_eq!(config.max_event_chars, None);
        assert_eq!(config.max_event_chars(), 8192);
        assert!(config.end_cues.is_empty());
        assert!(config.trust.is_empty());
        assert!(config.predicates.is_empty());
        assert_eq!(config.max_competing_per_key, 5);
        assert_eq!(config.min_confidence, 0.5);
        assert_eq!(config.llm_trust_factor, 0.8);
        assert_eq!(config.default_trust, 0.5);
    }

    #[test]
    fn max_event_chars_is_derived_from_the_context_unless_set() {
        // Arrange
        let derived = loaded_extraction("[extraction]\nctx_tokens = 1000\n");
        let set = loaded_extraction("[extraction]\nctx_tokens = 1000\nmax_event_chars = 77\n");

        // Act / Assert
        assert_eq!(derived.max_event_chars(), 2000);
        assert_eq!(set.max_event_chars(), 77);
    }

    #[test]
    fn numeric_keys_are_validated() {
        // Arrange: (toml line, whether the load must succeed)
        let rows = [
            ("max_events_per_run = 0", false),
            ("batch_events = 0", false),
            ("ctx_tokens = 0", false),
            ("request_timeout_secs = 0", false),
            ("start_timeout_secs = 0", false),
            ("run_deadline_secs = 0", false),
            ("max_competing_per_key = 0", false),
            ("max_batch_failures = 0", false),
            ("max_event_chars = 0", false),
            ("max_events_per_run = 1", true),
            ("batch_events = 1", true),
            ("ctx_tokens = 1", true),
            ("request_timeout_secs = 1", true),
            ("start_timeout_secs = 1", true),
            ("run_deadline_secs = 1", true),
            ("run_deadline_secs = 86400", true),
            ("run_deadline_secs = 86401", false),
            ("run_deadline_secs = 18446744073709551615", false),
            ("max_competing_per_key = 1", true),
            ("max_batch_failures = 1", true),
            ("max_event_chars = 1", true),
            ("port = 65536", false),
            ("port = 65535", true),
            ("min_confidence = -0.1", false),
            ("min_confidence = 1.1", false),
            ("min_confidence = nan", false),
            ("min_confidence = 0.0", true),
            ("min_confidence = 1.0", true),
            ("llm_trust_factor = 0.0", true),
            ("llm_trust_factor = 1.0", true),
            ("llm_trust_factor = 1.5", false),
            ("llm_trust_factor = nan", false),
            ("default_trust = -0.5", false),
            ("default_trust = 1.0", true),
        ];

        for (line, accepted) in rows {
            // Act
            let result = load(&format!("[extraction]\n{line}\n"));

            // Assert
            assert_eq!(result.is_ok(), accepted, "{line}: {result:?}");
        }
    }

    #[test]
    fn trust_scores_must_be_in_range() {
        // Arrange: (toml line, whether the load must succeed)
        let rows = [
            ("backfill = 0.0", true),
            ("backfill = 1.0", true),
            ("backfill = 1.1", false),
            ("notetaker = -0.1", false),
            ("notetaker = nan", false),
        ];

        for (line, accepted) in rows {
            // Act
            let result = load(&format!("[extraction.trust]\n{line}\n"));

            // Assert
            assert_eq!(result.is_ok(), accepted, "{line}: {result:?}");
        }
    }

    #[test]
    fn only_loopback_hosts_are_accepted() {
        // Arrange
        let rows = [
            ("127.0.0.1", true),
            ("127.0.0.2", true),
            ("::1", true),
            ("0.0.0.0", false),
            ("10.0.0.1", false),
            ("localhost", false),
            ("", false),
        ];

        for (host, accepted) in rows {
            // Act
            let result = load(&format!("[extraction]\nhost = \"{host}\"\n"));

            // Assert
            assert_eq!(result.is_ok(), accepted, "host {host:?}: {result:?}");
        }
    }

    #[test]
    fn extra_args_may_not_override_what_the_guard_sets() {
        // Arrange
        let rows: [(&str, bool); 9] = [
            (r#"["--host", "0.0.0.0"]"#, false),
            (r#"["--port=1"]"#, false),
            (r#"["--api-key", "x"]"#, false),
            (r#"["--api-key-file", "x"]"#, false),
            (r#"["--alias", "x"]"#, false),
            (r#"["-a", "x"]"#, false),
            (r#"["--threads", "4"]"#, true),
            (r#"["-t", "4"]"#, true),
            (r#"[]"#, true),
        ];

        for (args, accepted) in rows {
            // Act
            let result = load(&format!("[extraction]\nextra_args = {args}\n"));

            // Assert
            assert_eq!(result.is_ok(), accepted, "{args}: {result:?}");
        }
    }

    #[test]
    fn a_producer_id_may_not_collide_with_the_consolidation_tag() {
        // Arrange: (client id, whether the load must succeed)
        let rows = [
            ("consolidation", false),
            ("Consolidation", false),
            ("consolidation-bot", false),
            ("my-consolidation", true),
            ("notetaker", true),
        ];

        for (id, accepted) in rows {
            // Act
            let result = load(&format!("[producers.clients]\nsome-client = \"{id}\"\n"));

            // Assert
            assert_eq!(result.is_ok(), accepted, "{id}: {result:?}");
        }
    }

    #[test]
    fn the_unknown_producer_id_is_held_to_the_same_rule() {
        // Arrange / Act
        let message = error_of("[producers]\nunknown_id = \"consolidation-x\"\n");

        // Assert
        assert!(message.contains("consolidation"), "{message}");
    }

    #[test]
    fn a_cardinality_other_than_one_or_many_fails() {
        // Arrange
        let toml = "[extraction.predicates]\nworks_at = { cardinality = \"several\" }\n";

        // Act
        let message = error_of(toml);

        // Assert
        assert!(message.contains("several"), "{message}");
    }

    #[test]
    fn a_misspelled_key_inside_a_predicate_override_fails() {
        // Arrange
        let toml = "[extraction.predicates]\nworks_at = { cardnality = \"one\" }\n";

        // Act
        let message = error_of(toml);

        // Assert
        assert!(message.contains("cardnality"), "{message}");
    }

    #[test]
    fn a_huge_context_saturates_the_derived_event_bound() {
        // Arrange
        let config = loaded_extraction("[extraction]\nctx_tokens = 9223372036854775807\n");

        // Act
        let bound = config.max_event_chars();

        // Assert
        assert_eq!(bound, usize::MAX / 2);
    }

    #[test]
    fn built_in_predicates_hold_without_overrides() {
        // Arrange
        let vocabulary = resolve_vocabulary(&ExtractionConfig::default());

        // Act
        let lives_in = vocabulary.spec("lives_in");
        let born_on = vocabulary.spec("born_on");
        let knows = vocabulary.spec("knows");

        // Assert
        assert_eq!(lives_in.cardinality, Cardinality::One);
        assert_eq!(lives_in.match_mode, MatchMode::Contains);
        assert_eq!(born_on.cardinality, Cardinality::One);
        assert_eq!(born_on.grounding, Grounding::Date);
        assert_eq!(vocabulary.spec("works_at").match_mode, MatchMode::Exact);
        for single in ["has_title", "status"] {
            assert_eq!(vocabulary.spec(single).cardinality, Cardinality::One);
        }
        assert_eq!(knows.cardinality, Cardinality::Many);
        for rules_predicate in [
            "mentioned_in",
            "references_date",
            "references_url",
            "references_id",
            "member_of",
        ] {
            assert_eq!(
                vocabulary.spec(rules_predicate).cardinality,
                Cardinality::Many,
                "{rules_predicate}"
            );
        }
    }

    #[test]
    fn a_configured_predicate_overrides_the_built_in_and_keeps_its_other_fields() {
        // Arrange
        let config = loaded_extraction(
            "[extraction.predicates]\n\
             works_at = { cardinality = \"many\" }\n\
             lives_in = { match = \"exact\" }\n\
             speaks = { cardinality = \"one\", grounding = \"date\" }\n",
        );

        // Act
        let vocabulary = resolve_vocabulary(&config);

        // Assert
        assert_eq!(vocabulary.spec("works_at").cardinality, Cardinality::Many);
        let lives_in = vocabulary.spec("lives_in");
        assert_eq!(lives_in.match_mode, MatchMode::Exact);
        assert_eq!(lives_in.cardinality, Cardinality::One);
        let speaks = vocabulary.spec("speaks");
        assert_eq!(speaks.cardinality, Cardinality::One);
        assert_eq!(speaks.grounding, Grounding::Date);
        assert_eq!(speaks.match_mode, MatchMode::Exact);
    }

    #[test]
    fn an_unknown_predicate_resolves_to_many_with_exact_matching() {
        // Arrange
        let vocabulary = resolve_vocabulary(&ExtractionConfig::default());

        // Act
        let spec = vocabulary.spec("resides_in");

        // Assert
        assert_eq!(spec.cardinality, Cardinality::Many);
        assert_eq!(spec.match_mode, MatchMode::Exact);
    }

    #[test]
    fn declared_predicates_are_the_built_ins_plus_configured_names_sorted() {
        // Arrange
        let config = loaded_extraction("[extraction.predicates.speaks]\ncardinality = \"many\"\n");

        // Act
        let names = declared_predicates(&config);

        // Assert
        assert_eq!(names.len(), BUILT_IN_PREDICATES.len() + 1);
        assert!(names.contains(&"speaks".to_string()) && names.contains(&"works_at".to_string()));
        assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
    }
}
