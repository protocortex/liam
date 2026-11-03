// SPDX-License-Identifier: Apache-2.0
//! Deciding which lineages move between tiers, from one frozen snapshot.

use std::collections::HashSet;

use liam_store::{ItemView, LineageKey, Millis, Tier, UsageSnapshot};
use sha2::{Digest, Sha256};

use super::utility::{utility, TierParams};

/// Lineages that are never demoted because something else stands on them.
pub type ExemptSet = HashSet<LineageKey>;
/// Lineages a recall probe expects to find, which are never demoted.
pub type PinnedSet = HashSet<LineageKey>;

/// What earlier runs' rollbacks leave for this one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RollbackState {
    /// Runs in a row, up to this one, whose tier moves were rolled back.
    pub consecutive: u32,
    /// Lineages a rolled back run moved, with the runs they are left alone for.
    pub cooldown: Vec<(LineageKey, u32)>,
}

impl RollbackState {
    fn cooling_down(&self, key: &LineageKey) -> bool {
        self.cooldown
            .iter()
            .any(|(cooling, runs_left)| cooling == key && *runs_left > 0)
    }

    /// The cap on demotions: halved for each rollback in a row, never below 1,
    /// so a run that keeps failing the probes moves less each time.
    fn demotion_cap(&self, configured: usize) -> usize {
        let halved = configured.checked_shr(self.consecutive).unwrap_or(0);
        halved.max(1)
    }
}

/// Why a lineage is moving, as it is written into the log.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveReason {
    LowUtility,
    HighUtility,
    AccessedWhileCold,
}

impl MoveReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LowUtility => "low_utility",
            Self::HighUtility => "high_utility",
            Self::AccessedWhileCold => "accessed_while_cold",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannedMove {
    pub lineage_key: LineageKey,
    pub to: Tier,
    pub utility: f64,
    pub reason: MoveReason,
}

/// Candidates left out, by the first rule that left them out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SkipCounts {
    pub exempt: usize,
    pub pinned: usize,
    pub cooling_down: usize,
    pub resident: usize,
    /// No live node of the lineage has a vector, so there is nothing to drop.
    pub vectorless: usize,
    /// A node has a vector and no signature, so dropping it would lose it.
    pub unsigned: usize,
    /// Too little observed usage, or too young.
    pub not_judgeable: usize,
    pub over_promotion_cap: usize,
    pub over_demotion_cap: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TierPlan {
    pub promotions: Vec<PlannedMove>,
    pub demotions: Vec<PlannedMove>,
    pub skips: SkipCounts,
    /// Names the set of moves, so a rolled back run can be told from a repeat.
    pub plan_hash: String,
}

/// Plans the moves for `snapshot`. The same inputs give the same plan.
pub fn plan(
    snapshot: &UsageSnapshot,
    params: &TierParams,
    exempt: &ExemptSet,
    pinned: &PinnedSet,
    rollback: &RollbackState,
) -> TierPlan {
    let mut skips = SkipCounts::default();
    let mut promotions = Vec::new();
    let mut demotions = Vec::new();
    for item in &snapshot.items {
        let score = utility(item, params, snapshot.now).value;
        match item.tier {
            Tier::Cold => {
                if let Some(reason) = promotion_reason(item, score, params, snapshot.now) {
                    promotions.push(planned(item, Tier::Hot, score, reason));
                }
            }
            Tier::Hot if score < params.demote_threshold => {
                match demotion_block(item, snapshot, params, exempt, pinned, rollback) {
                    Some(block) => block.count(&mut skips),
                    None => {
                        demotions.push(planned(item, Tier::Cold, score, MoveReason::LowUtility))
                    }
                }
            }
            Tier::Hot => {}
        }
    }
    promotions.sort_by(|a, b| {
        b.utility
            .total_cmp(&a.utility)
            .then_with(|| a.lineage_key.cmp(&b.lineage_key))
    });
    demotions.sort_by(|a, b| {
        a.utility
            .total_cmp(&b.utility)
            .then_with(|| a.lineage_key.cmp(&b.lineage_key))
    });
    skips.over_promotion_cap = cut(&mut promotions, params.max_promotions);
    skips.over_demotion_cap = cut(&mut demotions, rollback.demotion_cap(params.max_demotions));
    let plan_hash = hash_moves(&promotions, &demotions);
    TierPlan {
        promotions,
        demotions,
        skips,
        plan_hash,
    }
}

fn planned(item: &ItemView, to: Tier, utility: f64, reason: MoveReason) -> PlannedMove {
    PlannedMove {
        lineage_key: item.lineage_key.clone(),
        to,
        utility,
        reason,
    }
}

/// A cold item promotes on its score, or on an access since it went cold once
/// its residency has passed, which lets a paraphrase hit bring it back before
/// its score does.
fn promotion_reason(
    item: &ItemView,
    score: f64,
    params: &TierParams,
    now: Millis,
) -> Option<MoveReason> {
    if score >= params.promote_threshold {
        return Some(MoveReason::HighUtility);
    }
    let accessed_since_cold = matches!(
        (item.last_access_at, item.changed_at),
        (Some(accessed), Some(changed)) if accessed > changed
    );
    let residency_over = item.residency_until.is_none_or(|until| until <= now);
    (accessed_since_cold && residency_over).then_some(MoveReason::AccessedWhileCold)
}

/// The rule that keeps a hot item with a low score where it is, if any.
#[derive(Clone, Copy)]
enum Block {
    Exempt,
    Pinned,
    CoolingDown,
    Resident,
    Vectorless,
    Unsigned,
    NotJudgeable,
}

impl Block {
    fn count(self, skips: &mut SkipCounts) {
        let counter = match self {
            Self::Exempt => &mut skips.exempt,
            Self::Pinned => &mut skips.pinned,
            Self::CoolingDown => &mut skips.cooling_down,
            Self::Resident => &mut skips.resident,
            Self::Vectorless => &mut skips.vectorless,
            Self::Unsigned => &mut skips.unsigned,
            Self::NotJudgeable => &mut skips.not_judgeable,
        };
        *counter += 1;
    }
}

fn demotion_block(
    item: &ItemView,
    snapshot: &UsageSnapshot,
    params: &TierParams,
    exempt: &ExemptSet,
    pinned: &PinnedSet,
    rollback: &RollbackState,
) -> Option<Block> {
    let key = &item.lineage_key;
    if exempt.contains(key) {
        Some(Block::Exempt)
    } else if pinned.contains(key) {
        Some(Block::Pinned)
    } else if rollback.cooling_down(key) {
        Some(Block::CoolingDown)
    } else if item
        .residency_until
        .is_some_and(|until| until > snapshot.now)
    {
        Some(Block::Resident)
    } else if !item.has_vector {
        Some(Block::Vectorless)
    } else if !item.has_signature {
        Some(Block::Unsigned)
    } else if !judgeable(item, snapshot, params) {
        Some(Block::NotJudgeable)
    } else {
        None
    }
}

/// An item can be judged unused only if usage was observed for a whole window
/// and the item has been around for the minimum age. Without the first, every
/// item of a store that never counted looks unused.
fn judgeable(item: &ItemView, snapshot: &UsageSnapshot, params: &TierParams) -> bool {
    let observed = snapshot.observation_since.is_some_and(|since| {
        snapshot.now.0 - since.0 >= Millis::days(i64::from(params.window_days)).0
    });
    let old_enough = snapshot.now.0 - item.newest_tx_from.0
        >= Millis::days(i64::from(params.min_item_age_days)).0;
    observed && old_enough
}

/// Keeps the first `cap` moves and says how many it dropped.
fn cut(moves: &mut Vec<PlannedMove>, cap: usize) -> usize {
    let dropped = moves.len().saturating_sub(cap);
    moves.truncate(cap);
    dropped
}

/// SHA-256 over the moves in key order, each as its target tier and its
/// length-prefixed key, as lowercase hex.
fn hash_moves(promotions: &[PlannedMove], demotions: &[PlannedMove]) -> String {
    let mut moves: Vec<&PlannedMove> = promotions.iter().chain(demotions).collect();
    moves.sort_by(|a, b| (a.to.as_str(), &a.lineage_key).cmp(&(b.to.as_str(), &b.lineage_key)));
    let mut hasher = Sha256::new();
    for planned in moves {
        let key = planned.lineage_key.as_str();
        hasher.update(planned.to.as_str().as_bytes());
        hasher.update((key.len() as u64).to_be_bytes());
        hasher.update(key.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use liam_store::{ItemView, LineageKey, Millis, Tier, UsageSnapshot};

    use super::super::utility::fixtures::*;
    use super::*;

    const DAY_MS: i64 = 86_400_000;
    /// Trust is the only weighted term in the threshold fixtures, so the score
    /// of an item is its producer's trust exactly.
    const ONLY_TRUST: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

    fn snapshot(items: Vec<ItemView>) -> UsageSnapshot {
        UsageSnapshot {
            now: NOW,
            producer_usage: Vec::new(),
            observation_since: Some(Millis(NOW.0 - 40 * DAY_MS)),
            items,
        }
    }

    /// Params whose score for `PRODUCER` items is `trust`.
    fn params(trust: f64) -> TierParams {
        let mut params = params_with(ONLY_TRUST, trust);
        params.promote_threshold = 0.6;
        params.demote_threshold = 0.2;
        params.window_days = 30;
        params.min_item_age_days = 14;
        params.max_promotions = 10;
        params.max_demotions = 10;
        params
    }

    fn plan_of(snapshot: &UsageSnapshot, params: &TierParams) -> TierPlan {
        plan(
            snapshot,
            params,
            &ExemptSet::new(),
            &PinnedSet::new(),
            &RollbackState::default(),
        )
    }

    fn cold(key: &str) -> ItemView {
        ItemView {
            tier: Tier::Cold,
            changed_at: Some(Millis(NOW.0 - 20 * DAY_MS)),
            has_vector: false,
            ..item(key)
        }
    }

    fn keys(moves: &[PlannedMove]) -> Vec<&str> {
        moves.iter().map(|m| m.lineage_key.as_str()).collect()
    }

    fn set(keys: &[&str]) -> HashSet<LineageKey> {
        keys.iter().map(|key| LineageKey::from_raw(*key)).collect()
    }

    #[test]
    fn promotion_fires_at_the_threshold_and_demotion_just_below_it() {
        let (promote, demote) = (params(0.0).promote_threshold, params(0.0).demote_threshold);
        for (name, subject, threshold, moves) in [
            ("promote", cold("subject"), promote, [false, true, true]),
            ("demote", item("subject"), demote, [true, false, false]),
        ] {
            for (trust, moves) in [threshold.next_down(), threshold, threshold.next_up()]
                .into_iter()
                .zip(moves)
            {
                // Act
                let planned = plan_of(&snapshot(vec![subject.clone()]), &params(trust));

                // Assert
                let moved = planned.promotions.len() + planned.demotions.len();
                assert_eq!(
                    moved == 1,
                    moves,
                    "{name} at {threshold} with trust {trust}"
                );
            }
        }
    }

    #[test]
    fn the_reasons_are_written_to_the_log_under_fixed_names() {
        // Act
        let names = [
            MoveReason::LowUtility,
            MoveReason::HighUtility,
            MoveReason::AccessedWhileCold,
        ]
        .map(MoveReason::as_str);

        // Assert
        assert_eq!(
            names,
            ["low_utility", "high_utility", "accessed_while_cold"]
        );
    }

    #[test]
    fn an_item_inside_the_band_stays_put() {
        // Arrange
        let items = vec![item("hot"), cold("cold")];

        // Act
        let planned = plan_of(&snapshot(items), &params(0.4));

        // Assert
        assert!(planned.promotions.is_empty() && planned.demotions.is_empty());
    }

    #[test]
    fn a_cold_item_accessed_since_it_went_cold_promotes_once_its_residency_is_over() {
        // Arrange: a score in the band, so only the access can move it
        let accessed = Some(Millis(NOW.0 - DAY_MS));
        let over = ItemView {
            last_access_at: accessed,
            residency_until: Some(NOW),
            ..cold("over")
        };
        let waiting = ItemView {
            last_access_at: accessed,
            residency_until: Some(Millis(NOW.0 + 1)),
            ..cold("waiting")
        };
        let unaccessed = ItemView {
            last_access_at: Some(Millis(NOW.0 - 30 * DAY_MS)),
            ..cold("unaccessed")
        };
        let none_yet = cold("none yet");

        // Act
        let planned = plan_of(
            &snapshot(vec![over, waiting, unaccessed, none_yet]),
            &params(0.4),
        );

        // Assert
        assert_eq!(keys(&planned.promotions), ["over"]);
        assert_eq!(planned.promotions[0].reason, MoveReason::AccessedWhileCold);
    }

    #[test]
    fn caps_keep_the_lowest_utility_demotions_and_the_highest_utility_promotions() {
        // Arrange: producers of different trust give items different scores
        let mut params = params(0.0);
        let mut hot_items = Vec::new();
        let mut cold_items = Vec::new();
        for (index, trust) in [0.05, 0.01, 0.1, 0.01].into_iter().enumerate() {
            let producer = format!("low-{index}");
            trusting(&mut params, &producer, trust);
            hot_items.push(ItemView {
                producer,
                ..item(&format!("hot-{index}"))
            });
        }
        for (index, trust) in [0.7, 0.9, 0.8, 0.9].into_iter().enumerate() {
            let producer = format!("high-{index}");
            trusting(&mut params, &producer, trust);
            cold_items.push(ItemView {
                producer,
                ..cold(&format!("cold-{index}"))
            });
        }
        params.max_demotions = 3;
        params.max_promotions = 2;
        hot_items.extend(cold_items);

        // Act
        let planned = plan_of(&snapshot(hot_items), &params);

        // Assert: the two equal scores break by key
        assert_eq!(keys(&planned.demotions), ["hot-1", "hot-3", "hot-0"]);
        assert_eq!(keys(&planned.promotions), ["cold-1", "cold-3"]);
        assert_eq!(planned.skips.over_demotion_cap, 1);
        assert_eq!(planned.skips.over_promotion_cap, 2);
    }

    #[test]
    fn each_consecutive_rollback_halves_the_demotion_cap_down_to_one() {
        for (configured, consecutive, expected) in [
            (1, 3, 1),
            (8, 0, 8),
            (8, 1, 4),
            (8, 2, 2),
            (8, 3, 1),
            (8, 40, 1),
        ] {
            // Arrange
            let mut params = params(0.0);
            params.max_demotions = configured;
            let items = (0..10).map(|i| item(&format!("k{i}"))).collect();
            let rollback = RollbackState {
                consecutive,
                cooldown: Vec::new(),
            };

            // Act
            let planned = plan(
                &snapshot(items),
                &params,
                &ExemptSet::new(),
                &PinnedSet::new(),
                &rollback,
            );

            // Assert
            assert_eq!(
                planned.demotions.len(),
                expected,
                "{configured} after {consecutive}"
            );
        }
    }

    #[test]
    fn an_item_something_protects_is_not_demoted_and_the_unembedded_do_not_use_the_cap() {
        // Arrange
        let mut params = params(0.0);
        params.max_demotions = 1;
        let items = vec![
            item("a-entity"),
            item("b-pinned"),
            item("c-cooling"),
            ItemView {
                residency_until: Some(Millis(NOW.0 + 1)),
                ..item("d-resident")
            },
            ItemView {
                has_vector: false,
                ..item("e-vectorless")
            },
            ItemView {
                has_signature: false,
                ..item("f-unsigned")
            },
            item("g-free"),
        ];
        let rollback = RollbackState {
            consecutive: 0,
            cooldown: vec![
                (LineageKey::from_raw("c-cooling"), 2),
                (LineageKey::from_raw("g-free"), 0),
            ],
        };

        // Act
        let planned = plan(
            &snapshot(items),
            &params,
            &set(&["a-entity"]),
            &set(&["b-pinned"]),
            &rollback,
        );

        // Assert: a spent cooldown (0 runs left) no longer protects
        assert_eq!(keys(&planned.demotions), ["g-free"]);
        assert_eq!(
            planned.skips,
            SkipCounts {
                exempt: 1,
                pinned: 1,
                cooling_down: 1,
                resident: 1,
                vectorless: 1,
                unsigned: 1,
                ..SkipCounts::default()
            }
        );
    }

    #[test]
    fn an_item_is_judged_only_after_a_full_window_of_observation_and_the_minimum_age() {
        let old = item("old");
        let young = ItemView {
            newest_tx_from: Millis(NOW.0 - 14 * DAY_MS + 1),
            ..item("young")
        };
        let just_old_enough = ItemView {
            newest_tx_from: Millis(NOW.0 - 14 * DAY_MS),
            ..item("just old enough")
        };
        for (observed_days, items, demoted) in [
            (Some(30), vec![old.clone()], vec!["old"]),
            (Some(29), vec![old.clone()], vec![]),
            (None, vec![old.clone()], vec![]),
            (
                Some(60),
                vec![young, just_old_enough],
                vec!["just old enough"],
            ),
        ] {
            // Arrange
            let mut snap = snapshot(items);
            snap.observation_since = observed_days.map(|days| Millis(NOW.0 - days * DAY_MS));

            // Act
            let planned = plan_of(&snap, &params(0.0));

            // Assert
            assert_eq!(keys(&planned.demotions), demoted, "{observed_days:?}");
        }
    }

    #[test]
    fn a_node_written_now_with_a_backdated_valid_from_is_too_young() {
        // Arrange: the snapshot carries write time only, so a backdated
        // valid_from has no way in; a write time of now is young
        let fresh = ItemView {
            newest_tx_from: NOW,
            ..item("fresh")
        };

        // Act
        let planned = plan_of(&snapshot(vec![fresh]), &params(0.0));

        // Assert
        assert!(planned.demotions.is_empty());
        assert_eq!(planned.skips.not_judgeable, 1);
    }

    #[test]
    fn the_same_snapshot_gives_the_same_plan_and_hash_whatever_the_item_order() {
        // Arrange
        let items: Vec<ItemView> = (0..6).map(|i| item(&format!("k{i}"))).collect();
        let mut reversed = items.clone();
        reversed.reverse();

        // Act
        let first = plan_of(&snapshot(items.clone()), &params(0.0));
        let again = plan_of(&snapshot(items), &params(0.0));
        let other_order = plan_of(&snapshot(reversed), &params(0.0));

        // Assert
        assert_eq!(first, again);
        assert_eq!(first.plan_hash, other_order.plan_hash);
        assert_eq!(first.plan_hash.len(), 64);
    }

    #[test]
    fn a_different_set_of_moves_has_a_different_hash() {
        // Arrange
        let both: Vec<ItemView> = vec![item("a"), item("b")];
        let one: Vec<ItemView> = vec![item("a")];

        // Act
        let (with_both, with_one) = (
            plan_of(&snapshot(both), &params(0.0)),
            plan_of(&snapshot(one), &params(0.0)),
        );

        // Assert
        assert_ne!(with_both.plan_hash, with_one.plan_hash);
    }
}
