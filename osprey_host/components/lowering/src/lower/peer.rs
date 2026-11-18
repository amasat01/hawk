//! The record that lets several libraries lower one build to the same cut: what a library states of its own decisions
//! (`record`), how a library that lowers beside it takes them over (`lay_over`, `fusion_row`) and the check that its units are the
//! units the leader named (`check_cut`).
use super::card::BuildCard;
use super::request::KernelRequest;
use super::unit_meta::{PeerFusion, PeerNumbers, PeerRecord, UnitMeta};
use crate::facts::FusionRow;
use aether_writers::refuse::{bad, Refusal};

/// The numbers the steps of a lowering decide on, as a card states them.
fn numbers_of(card: &BuildCard) -> PeerNumbers {
    PeerNumbers {
        rolled_fold_min: card.rolled_fold_min,
        linalg_max_dim: card.linalg_max_dim,
        linalg_team_max_dim: card.linalg_team_max_dim,
        copy_threads: card.copy_threads,
        team_threads: card.team_threads,
        team_threads_large: card.team_threads_large,
        scatter_max_chunks: card.scatter_max_chunks,
        scatter_threads: card.scatter_threads,
        cells_row: card.cells_row,
    }
}

/// The card a library decides on when it lowers beside a leader: its own card with the leader's numbers over it. The rows of
/// `steps_reverse` stay the library's own: each states the rows of its target.
pub(super) fn lay_over(card: BuildCard, peer: &PeerNumbers) -> BuildCard {
    BuildCard {
        rolled_fold_min: peer.rolled_fold_min,
        linalg_max_dim: peer.linalg_max_dim,
        linalg_team_max_dim: peer.linalg_team_max_dim,
        copy_threads: peer.copy_threads,
        team_threads: peer.team_threads,
        team_threads_large: peer.team_threads_large,
        scatter_max_chunks: peer.scatter_max_chunks,
        scatter_threads: peer.scatter_threads,
        cells_row: peer.cells_row,
        ..card
    }
}

/// The fusion row a leader stated, as the chain step reads it.
pub(super) fn fusion_row(row: &PeerFusion) -> FusionRow {
    FusionRow {
        regs_base: row.regs_base,
        regs_file: row.regs_file,
        launch_us: row.launch_us,
        plane_us_per_byte: row.plane_us_per_byte,
        local_over_plane: row.local_over_plane,
    }
}

/// The record a library states of a lowering it has done: the numbers it decided on, the fusion row of a chain (none for another
/// body), whether its contraction runs directly, and the names of its units.
pub(super) fn record(
    deciding: &BuildCard,
    fusion: Option<&FusionRow>,
    direct_contraction: bool,
    units: &[&UnitMeta],
) -> PeerRecord {
    PeerRecord {
        numbers: numbers_of(deciding),
        fusion: fusion.map(|row| PeerFusion {
            regs_base: row.regs_base,
            regs_file: row.regs_file,
            launch_us: row.launch_us,
            plane_us_per_byte: row.plane_us_per_byte,
            local_over_plane: row.local_over_plane,
        }),
        direct_contraction,
        units: units.iter().map(|u| u.name.clone()).collect(),
    }
}

/// Refuses a lowering whose units are not the units the leader named: this library cannot meet the leader's cut.
pub(super) fn check_cut(request: &KernelRequest, units: &[&UnitMeta]) -> Result<(), Refusal> {
    let Some((leader, peer)) = &request.peer else {
        return Ok(());
    };
    let ours: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
    match ours == peer.units {
        true => Ok(()),
        false => Err(bad(format!(
            "{:?}: the cut of the {leader:?} library is the units {:?}; this library lowers the units {ours:?}",
            request.name, peer.units
        ))),
    }
}
