"""
CoverGuard 4-satellite downlink scheduler simulation.

Assumptions made explicit:
1. Simulation horizon is 72 hours; onboard data time-to-live is 14 days.
   Expiration logic is implemented even though the default horizon is shorter than 14 days.
2. Four satellites receive the same generated mission scenario for every scheduler.
   Contacts, packet creation times, packet sizes, parcel health, confidence, novelty,
   and raw-data requests are identical across schedulers for a given seed.
3. Raw-data requests are indivisible full products only. No preview or medium products
   are generated, scored, transmitted, or counted. A raw request is completed only when
   the full raw_data packet is downlinked.
4. Fault packets are compact telemetry products. Raw packets are simulated as compressed
   multi-band image products plus 13 metadata/statistics fields.
5. FIFO baselines consider packets in FIFO order but skip an indivisible packet if it
   does not fit in the remaining capacity of the current pass. This avoids making the
   baselines artificially bad due only to head-of-line blocking from one oversized raw
   packet, while still giving them no class/utility optimization.
6. Adaptive scheduling uses an exact 0/1 knapsack over all eligible packets for the
   contacted satellite at each ground pass, maximizing total decision utility subject
   to pass capacity.
7. Adaptive stale suppression removes older regular faults when a newer fault state for
   the same parcel is known onboard. Emergency fault packets are retained as alert/audit
   products; their utility is still penalized when a newer state exists.

Run:
    python coverguard_scheduler_sim.py

The script prints a detailed single-seed summary followed by Monte Carlo medians.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EMERGENCY = "emergency_fault"
REGULAR = "regular_fault"
RAW = "raw_data"
RAW_FULL = "raw_data_full"

RAW_STATS = (
    "NDVI",
    "NDMI",
    "NDRE",
    "SAVI",
    "MSAVI",
    "bare_soil_fraction",
    "patchiness",
    "edge_concentration",
    "qa_score",
    "canopy_cover",
    "thermal_anomaly",
    "chlorophyll_proxy",
    "water_stress_index",
)

SCHEDULERS = ("naive_fifo", "priority_fifo", "adaptive")


@dataclass(frozen=True)
class SimConfig:
    n_satellites: int = 4
    sim_hours: float = 72.0
    storage_ttl_hours: float = 14.0 * 24.0

    n_parcels: int = 160
    mean_observations_per_sat: int = 240
    mean_raw_requests_per_sat: int = 13
    high_novelty_threshold: float = 0.86

    # Ground-pass model. Capacity is deliberately constrained to represent only
    # the science-data allocation after link overhead, housekeeping, and pass losses.
    mean_contact_gap_h: float = 5.5
    contact_gap_jitter_h: float = 1.1
    min_contact_gap_h: float = 2.8

    # Utility constants. These are dimensionless decision-utility units.
    emergency_class_value: float = 2400.0
    regular_class_value: float = 520.0
    raw_full_completion_value: float = 32000.0

    emergency_tau_h: float = 7.0   # emergency utility decays quickly
    regular_tau_h: float = 22.0    # regular fault utility decays more slowly
    raw_tau_h: float = 72.0        # raw review utility decays slowly

    regular_stale_fault_factor: float = 0.05
    emergency_stale_fault_factor: float = 0.20


# -----------------------------------------------------------------------------
# Core data records
# -----------------------------------------------------------------------------

@dataclass
class Contact:
    contact_id: int
    sat_id: int
    start_h: float
    duration_min: float
    data_rate_mbps: float
    science_fraction: float
    capacity_kb: int


@dataclass
class Packet:
    packet_id: int
    sat_id: int
    created_at_h: float
    packet_class: str
    size_kb: int
    parcel_id: int
    location: Tuple[float, float]
    field_health_score: float
    fault_cause: str
    confidence: float
    novelty: float
    timestamp_h: float

    # Raw-data-only fields. They remain None/empty for fault packets.
    raw_request_id: Optional[int] = None
    image_pixels_x: Optional[int] = None
    image_pixels_y: Optional[int] = None
    image_bands: Optional[int] = None
    bytes_per_sample: Optional[int] = None
    compression_factor: Optional[float] = None
    raw_stats: Dict[str, float] = field(default_factory=dict)

    # Explicit raw state tracking. Preview and medium remain False by design.
    preview_sent: bool = False
    medium_sent: bool = False
    full_raw_sent: bool = False
    raw_request_completed: bool = False


@dataclass
class SchedulerResult:
    scheduler: str
    deliveries: List[Dict]
    stale_removed: int
    expired_removed: int
    total_capacity_kb: int


# -----------------------------------------------------------------------------
# Scenario generation
# -----------------------------------------------------------------------------

def generate_parcels(rng: np.random.Generator, cfg: SimConfig) -> pd.DataFrame:
    """Create persistent parcel properties used by every satellite and scheduler."""
    parcel_id = np.arange(cfg.n_parcels)

    # A synthetic agricultural region. Location values are only for packet metadata.
    lat = rng.uniform(34.5, 41.5, cfg.n_parcels)
    lon = rng.uniform(-123.5, -115.0, cfg.n_parcels)

    base_health = rng.normal(loc=6.5, scale=1.5, size=cfg.n_parcels)
    stressed = rng.random(cfg.n_parcels) < 0.20
    base_health[stressed] -= rng.uniform(1.6, 3.8, stressed.sum())
    base_health = np.clip(base_health, 0.2, 9.8)

    base_novelty = rng.beta(a=1.2, b=5.0, size=cfg.n_parcels)
    seasonal_phase = rng.uniform(0.0, 2.0 * math.pi, cfg.n_parcels)
    drift_per_hour = rng.normal(loc=-0.002, scale=0.009, size=cfg.n_parcels)

    shock_start = rng.uniform(6.0, 56.0, cfg.n_parcels)
    shock_magnitude = np.where(
        rng.random(cfg.n_parcels) < 0.18,
        rng.uniform(0.8, 3.2, cfg.n_parcels),
        0.0,
    )

    return pd.DataFrame(
        {
            "parcel_id": parcel_id,
            "lat": lat,
            "lon": lon,
            "base_health": base_health,
            "base_novelty": base_novelty,
            "seasonal_phase": seasonal_phase,
            "drift_per_hour": drift_per_hour,
            "shock_start_h": shock_start,
            "shock_magnitude": shock_magnitude,
        }
    )


def sample_field_state(
    rng: np.random.Generator,
    parcel: pd.Series,
    t_h: float,
) -> Tuple[float, float, float]:
    """Return field health score, novelty, and model confidence at a time."""
    diurnal = 0.35 * math.sin(2.0 * math.pi * t_h / 24.0 + float(parcel.seasonal_phase))
    drift = float(parcel.drift_per_hour) * t_h
    shock = float(parcel.shock_magnitude) if t_h >= float(parcel.shock_start_h) else 0.0
    health = float(parcel.base_health) + diurnal + drift - shock + rng.normal(0.0, 0.8)
    health = float(np.clip(health, 0.0, 10.0))

    novelty_burst = rng.beta(0.8, 5.5)
    if rng.random() < 0.035:
        novelty_burst += rng.uniform(0.35, 0.75)
    novelty = float(np.clip(float(parcel.base_novelty) + novelty_burst, 0.0, 1.0))

    # Lower QA / ambiguous scenes reduce confidence, but most model outputs are usable.
    confidence = float(np.clip(rng.normal(0.84, 0.10), 0.50, 0.99))
    return health, novelty, confidence


def choose_fault_cause(rng: np.random.Generator, health: float, novelty: float) -> str:
    if novelty >= 0.86 and health >= 5.0:
        return "novel_pattern"
    if health < 3.0:
        causes = ["acute_water_stress", "pest_or_disease", "sensor_qa_alarm", "flooding_or_waterlogging"]
        probs = [0.40, 0.30, 0.10, 0.20]
    else:
        causes = ["water_stress", "nutrient_deficiency", "patchy_growth", "pest_or_disease", "sensor_qa_warning"]
        probs = [0.32, 0.22, 0.22, 0.16, 0.08]
    return str(rng.choice(causes, p=probs))


def make_raw_stats(rng: np.random.Generator, health: float, novelty: float, confidence: float) -> Dict[str, float]:
    """Simulate the 13 requested raw-data statistics/metadata fields."""
    severity = (10.0 - health) / 10.0
    stats = {
        "NDVI": np.clip(0.75 - 0.55 * severity + rng.normal(0.0, 0.04), -0.2, 0.95),
        "NDMI": np.clip(0.45 - 0.60 * severity + rng.normal(0.0, 0.05), -0.4, 0.8),
        "NDRE": np.clip(0.55 - 0.35 * severity + rng.normal(0.0, 0.04), -0.2, 0.9),
        "SAVI": np.clip(0.65 - 0.45 * severity + rng.normal(0.0, 0.04), -0.2, 0.95),
        "MSAVI": np.clip(0.62 - 0.44 * severity + rng.normal(0.0, 0.04), -0.2, 0.95),
        "bare_soil_fraction": np.clip(0.10 + 0.50 * severity + rng.normal(0.0, 0.04), 0.0, 1.0),
        "patchiness": np.clip(0.15 + 0.55 * severity + 0.20 * novelty + rng.normal(0.0, 0.05), 0.0, 1.0),
        "edge_concentration": np.clip(0.10 + 0.35 * novelty + rng.normal(0.0, 0.06), 0.0, 1.0),
        "qa_score": confidence,
        "canopy_cover": np.clip(0.82 - 0.62 * severity + rng.normal(0.0, 0.05), 0.0, 1.0),
        "thermal_anomaly": np.clip(0.08 + 0.75 * severity + rng.normal(0.0, 0.08), 0.0, 1.0),
        "chlorophyll_proxy": np.clip(0.72 - 0.52 * severity + rng.normal(0.0, 0.05), 0.0, 1.0),
        "water_stress_index": np.clip(0.15 + 0.75 * severity + rng.normal(0.0, 0.05), 0.0, 1.0),
    }
    return {k: float(v) for k, v in stats.items()}


def simulate_raw_size_kb(rng: np.random.Generator) -> Tuple[int, int, int, int, float, int]:
    """
    Simulate a compressed multi-band image product size.

    Returns:
        pixels_x, pixels_y, bands, bytes_per_sample, compression_factor, size_kb
    """
    # Small parcel-level chips, not full scenes. Compression factor is the retained
    # fraction of the uncompressed data size after image compression.
    pixels_x = int(rng.choice([384, 448, 512, 640, 768], p=[0.16, 0.22, 0.30, 0.22, 0.10]))
    pixels_y = int(rng.choice([384, 448, 512, 640, 768], p=[0.16, 0.22, 0.30, 0.22, 0.10]))
    bands = int(rng.choice([8, 10, 12, 13], p=[0.20, 0.30, 0.30, 0.20]))
    bytes_per_sample = 2
    compression_factor = float(rng.uniform(0.22, 0.48))
    stats_metadata_kb = 6  # 13 stats plus small headers, timestamps, geotags, QA metadata.
    uncompressed_bytes = pixels_x * pixels_y * bands * bytes_per_sample
    size_kb = int(math.ceil((uncompressed_bytes * compression_factor) / 1024.0 + stats_metadata_kb))
    return pixels_x, pixels_y, bands, bytes_per_sample, compression_factor, size_kb


def generate_packets(rng: np.random.Generator, parcels: pd.DataFrame, cfg: SimConfig) -> List[Packet]:
    packets: List[Packet] = []
    packet_id = 0
    raw_request_id = 0

    parcel_rows = [row for _, row in parcels.iterrows()]
    parcel_ids = parcels["parcel_id"].to_numpy()

    # Satellite-specific preferred coverage areas create repeated parcel observations,
    # which makes staleness realistic without changing the scenario by scheduler.
    sat_primary_sets = []
    for sat_id in range(cfg.n_satellites):
        start = int((sat_id / cfg.n_satellites) * cfg.n_parcels)
        stop = int(((sat_id + 1) / cfg.n_satellites) * cfg.n_parcels)
        sat_primary_sets.append(parcel_ids[start:stop])

    for sat_id in range(cfg.n_satellites):
        n_obs = int(rng.poisson(cfg.mean_observations_per_sat))
        obs_times = np.sort(rng.uniform(0.0, cfg.sim_hours, n_obs))

        for t_h in obs_times:
            if rng.random() < 0.68:
                parcel_id = int(rng.choice(sat_primary_sets[sat_id]))
            else:
                parcel_id = int(rng.choice(parcel_ids))

            parcel = parcel_rows[parcel_id]
            health, novelty, confidence = sample_field_state(rng, parcel, float(t_h))

            if health < 3.0:
                packet_class = EMERGENCY
                size_kb = int(rng.integers(2, 7))  # 2--6 KB inclusive
            elif health < 5.0 or novelty >= cfg.high_novelty_threshold:
                packet_class = REGULAR
                size_kb = int(rng.integers(6, 19))  # 6--18 KB inclusive
            else:
                continue

            packets.append(
                Packet(
                    packet_id=packet_id,
                    sat_id=sat_id,
                    created_at_h=float(t_h),
                    packet_class=packet_class,
                    size_kb=size_kb,
                    parcel_id=parcel_id,
                    location=(float(parcel.lat), float(parcel.lon)),
                    field_health_score=health,
                    fault_cause=choose_fault_cause(rng, health, novelty),
                    confidence=confidence,
                    novelty=novelty,
                    timestamp_h=float(t_h),
                )
            )
            packet_id += 1

        # Explicit random raw-data requests from the ground.
        n_raw = int(rng.poisson(cfg.mean_raw_requests_per_sat))
        request_times = np.sort(rng.uniform(0.0, cfg.sim_hours, n_raw))

        for t_h in request_times:
            if rng.random() < 0.62:
                parcel_id = int(rng.choice(sat_primary_sets[sat_id]))
            else:
                parcel_id = int(rng.choice(parcel_ids))
            parcel = parcel_rows[parcel_id]
            health, novelty, confidence = sample_field_state(rng, parcel, float(t_h))
            px, py, bands, bps, compression, size_kb = simulate_raw_size_kb(rng)

            packets.append(
                Packet(
                    packet_id=packet_id,
                    sat_id=sat_id,
                    created_at_h=float(t_h),
                    packet_class=RAW,
                    size_kb=size_kb,
                    parcel_id=parcel_id,
                    location=(float(parcel.lat), float(parcel.lon)),
                    field_health_score=health,
                    fault_cause="ground_requested_raw_review",
                    confidence=confidence,
                    novelty=novelty,
                    timestamp_h=float(t_h),
                    raw_request_id=raw_request_id,
                    image_pixels_x=px,
                    image_pixels_y=py,
                    image_bands=bands,
                    bytes_per_sample=bps,
                    compression_factor=compression,
                    raw_stats=make_raw_stats(rng, health, novelty, confidence),
                )
            )
            packet_id += 1
            raw_request_id += 1

    return sorted(packets, key=lambda p: (p.created_at_h, p.packet_id))


def generate_contacts(rng: np.random.Generator, cfg: SimConfig) -> List[Contact]:
    contacts: List[Contact] = []
    contact_id = 0

    for sat_id in range(cfg.n_satellites):
        t_h = float(rng.uniform(0.35, 1.9))
        while t_h < cfg.sim_hours:
            duration_min = float(rng.triangular(left=5.5, mode=8.5, right=13.5))
            data_rate_mbps = float(np.clip(rng.lognormal(mean=math.log(0.075), sigma=0.38), 0.035, 0.16))
            science_fraction = float(rng.uniform(0.25, 0.60))

            # 1 Mbps = 125 KB/s. Capacity is the science-data share of a pass.
            capacity_kb = int(max(1, math.floor(duration_min * 60.0 * data_rate_mbps * 125.0 * science_fraction)))

            contacts.append(
                Contact(
                    contact_id=contact_id,
                    sat_id=sat_id,
                    start_h=t_h,
                    duration_min=duration_min,
                    data_rate_mbps=data_rate_mbps,
                    science_fraction=science_fraction,
                    capacity_kb=capacity_kb,
                )
            )
            contact_id += 1
            gap_h = max(cfg.min_contact_gap_h, float(rng.normal(cfg.mean_contact_gap_h, cfg.contact_gap_jitter_h)))
            t_h += gap_h

    return sorted(contacts, key=lambda c: (c.start_h, c.sat_id, c.contact_id))


def build_scenario(seed: int, cfg: SimConfig) -> Tuple[List[Packet], List[Contact]]:
    """Build one fixed scenario that every scheduler will receive identically."""
    rng = np.random.default_rng(seed)
    parcels = generate_parcels(rng, cfg)
    packets = generate_packets(rng, parcels, cfg)
    contacts = generate_contacts(rng, cfg)
    return packets, contacts


# -----------------------------------------------------------------------------
# Utility model
# -----------------------------------------------------------------------------

def product_class(packet: Packet) -> str:
    return RAW_FULL if packet.packet_class == RAW else packet.packet_class


def is_fault(packet: Packet) -> bool:
    return packet.packet_class in (EMERGENCY, REGULAR)


def field_severity(packet: Packet) -> float:
    return (10.0 - packet.field_health_score) / 10.0


def fault_content_multiplier(packet: Packet) -> float:
    """
    Content multiplier for faults:
        severity = (10 - Field Health Score) / 10
        confidence = model confidence
        novelty = parcel novelty

    The constant term prevents moderate but confident faults from going to zero;
    severity, confidence, and novelty all increase decision value.
    """
    severity = field_severity(packet)
    return 0.50 + 1.15 * severity + 0.65 * packet.confidence + 0.80 * packet.novelty


def raw_content_multiplier(packet: Packet) -> float:
    """
    Raw full-product completion utility values novelty, confidence, and severity.
    It is still completion-only: no partial preview or medium utility is available.
    """
    severity = field_severity(packet)
    return 0.70 + 0.65 * packet.novelty + 0.30 * severity + 0.25 * packet.confidence


def packet_utility(
    packet: Packet,
    delivered_at_h: float,
    cfg: SimConfig,
    latest_fault_time_by_parcel: Dict[int, float],
) -> float:
    """Decision utility used for all schedulers and for adaptive knapsack scoring."""
    age_h = max(0.0, delivered_at_h - packet.created_at_h)

    if packet.packet_class == EMERGENCY:
        class_value = cfg.emergency_class_value
        latency_decay = math.exp(-age_h / cfg.emergency_tau_h)
        stale_factor = cfg.emergency_stale_fault_factor if latest_fault_time_by_parcel.get(packet.parcel_id, packet.created_at_h) > packet.created_at_h else 1.0
        return class_value * fault_content_multiplier(packet) * latency_decay * stale_factor

    if packet.packet_class == REGULAR:
        class_value = cfg.regular_class_value
        latency_decay = math.exp(-age_h / cfg.regular_tau_h)
        stale_factor = cfg.regular_stale_fault_factor if latest_fault_time_by_parcel.get(packet.parcel_id, packet.created_at_h) > packet.created_at_h else 1.0
        return class_value * fault_content_multiplier(packet) * latency_decay * stale_factor

    if packet.packet_class == RAW:
        # Full raw completion utility only. This is intentionally not a partial
        # preview value. raw_request_completed becomes True only when this packet is sent.
        latency_decay = math.exp(-age_h / cfg.raw_tau_h)
        return cfg.raw_full_completion_value * raw_content_multiplier(packet) * latency_decay

    raise ValueError(f"Unknown packet class: {packet.packet_class}")


# -----------------------------------------------------------------------------
# Scheduling algorithms
# -----------------------------------------------------------------------------

def fifo_select(queue: Sequence[Packet], capacity_kb: int, priority_aware: bool) -> List[Packet]:
    """
    Select indivisible packets in FIFO order, optionally with class priority.

    Packets that cannot fit in the remaining pass capacity are left queued; the scan
    continues so a single large raw packet does not create pure head-of-line blocking.
    """
    if priority_aware:
        class_rank = {EMERGENCY: 0, REGULAR: 1, RAW: 2}
        ordered = sorted(queue, key=lambda p: (class_rank[p.packet_class], p.created_at_h, p.packet_id))
    else:
        ordered = sorted(queue, key=lambda p: (p.created_at_h, p.packet_id))

    selected: List[Packet] = []
    remaining = capacity_kb
    for packet in ordered:
        if packet.size_kb <= remaining:
            selected.append(packet)
            remaining -= packet.size_kb
    return selected


def knapsack_select(items: Sequence[Packet], utilities: Sequence[float], capacity_kb: int) -> List[Packet]:
    """Exact 0/1 knapsack: maximize total utility subject to sum(size_kb) <= capacity."""
    if capacity_kb <= 0 or len(items) == 0:
        return []

    weights_all = np.array([max(1, int(item.size_kb)) for item in items], dtype=np.int32)
    values_all = np.array(utilities, dtype=np.float64)
    valid_mask = (weights_all <= capacity_kb) & (values_all > 0.0)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) == 0:
        return []

    weights = weights_all[valid_indices]
    values = values_all[valid_indices]
    n = len(valid_indices)

    dp = np.zeros(capacity_kb + 1, dtype=np.float64)
    keep = np.zeros((n, capacity_kb + 1), dtype=np.bool_)

    for i, (w, v) in enumerate(zip(weights, values)):
        candidate = dp[:-w] + v
        better = candidate > dp[w:] + 1e-12
        if np.any(better):
            capacities = np.flatnonzero(better) + w
            keep[i, capacities] = True
            dp[capacities] = candidate[better]

    cap = int(np.argmax(dp))
    selected_indices: List[int] = []
    for i in range(n - 1, -1, -1):
        if keep[i, cap]:
            selected_indices.append(int(valid_indices[i]))
            cap -= int(weights[i])

    selected_indices.reverse()
    return [items[i] for i in selected_indices]


def suppress_stale_faults(
    queue: Sequence[Packet],
    latest_fault_time_by_parcel: Dict[int, float],
) -> Tuple[List[Packet], int]:
    """
    Adaptive stale suppression.

    Older regular faults are removed if any newer fault state for the same parcel exists.
    Emergency fault packets are retained as alert/audit products, although their utility
    is still stale-penalized if a newer state exists. Raw products are never removed as
    stale because a raw request is an explicit ground task.
    """
    kept: List[Packet] = []
    removed = 0

    for packet in queue:
        if packet.packet_class == REGULAR:
            if latest_fault_time_by_parcel.get(packet.parcel_id, packet.created_at_h) > packet.created_at_h:
                removed += 1
                continue
        # Emergency faults are retained as alert/audit packets; their utility still
        # receives an emergency stale penalty if a newer state exists.
        kept.append(packet)

    return kept, removed


def run_scheduler(
    scheduler: str,
    packets: Sequence[Packet],
    contacts: Sequence[Contact],
    cfg: SimConfig,
) -> SchedulerResult:
    if scheduler not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    # Deep copy prevents one scheduler's raw completion flags from affecting another.
    stream = sorted(copy.deepcopy(list(packets)), key=lambda p: (p.created_at_h, p.packet_id))
    contacts_sorted = sorted(list(contacts), key=lambda c: (c.start_h, c.sat_id, c.contact_id))

    onboard: Dict[int, List[Packet]] = {sat_id: [] for sat_id in range(cfg.n_satellites)}
    latest_fault_time: Dict[int, Dict[int, float]] = {sat_id: {} for sat_id in range(cfg.n_satellites)}

    deliveries: List[Dict] = []
    stale_removed = 0
    expired_removed = 0
    next_packet_idx = 0
    total_capacity_kb = int(sum(c.capacity_kb for c in contacts_sorted))

    for contact in contacts_sorted:
        # Add all packets created up to this time to the appropriate satellite queue.
        while next_packet_idx < len(stream) and stream[next_packet_idx].created_at_h <= contact.start_h:
            packet = stream[next_packet_idx]
            onboard[packet.sat_id].append(packet)

            if is_fault(packet):
                current_latest = latest_fault_time[packet.sat_id].get(packet.parcel_id, -math.inf)
                if packet.created_at_h > current_latest:
                    latest_fault_time[packet.sat_id][packet.parcel_id] = packet.created_at_h
            next_packet_idx += 1

        sat_queue = onboard[contact.sat_id]

        # Enforce 14-day onboard storage expiration.
        unexpired: List[Packet] = []
        for packet in sat_queue:
            if contact.start_h - packet.created_at_h <= cfg.storage_ttl_hours:
                unexpired.append(packet)
            else:
                expired_removed += 1
        sat_queue = unexpired

        if scheduler == "adaptive":
            sat_queue, removed_now = suppress_stale_faults(
                sat_queue,
                latest_fault_time[contact.sat_id],
            )
            stale_removed += removed_now

        if scheduler == "naive_fifo":
            selected = fifo_select(sat_queue, contact.capacity_kb, priority_aware=False)
        elif scheduler == "priority_fifo":
            selected = fifo_select(sat_queue, contact.capacity_kb, priority_aware=True)
        else:
            utilities = [
                packet_utility(packet, contact.start_h, cfg, latest_fault_time[contact.sat_id])
                for packet in sat_queue
            ]
            selected = knapsack_select(sat_queue, utilities, contact.capacity_kb)

        selected_ids = {packet.packet_id for packet in selected}

        for packet in selected:
            if packet.packet_class == RAW:
                packet.full_raw_sent = True
                packet.raw_request_completed = True
                # These remain false by design; kept explicit for auditability.
                packet.preview_sent = False
                packet.medium_sent = False

            utility = packet_utility(packet, contact.start_h, cfg, latest_fault_time[contact.sat_id])
            deliveries.append(
                {
                    "scheduler": scheduler,
                    "contact_id": contact.contact_id,
                    "sat_id": contact.sat_id,
                    "packet_id": packet.packet_id,
                    "packet_class": packet.packet_class,
                    "product_class": product_class(packet),
                    "parcel_id": packet.parcel_id,
                    "created_at_h": packet.created_at_h,
                    "delivered_at_h": contact.start_h,
                    "latency_h": contact.start_h - packet.created_at_h,
                    "size_kb": packet.size_kb,
                    "utility": utility,
                    "raw_request_completed": packet.raw_request_completed,
                    "full_raw_sent": packet.full_raw_sent,
                    "preview_sent": packet.preview_sent,
                    "medium_sent": packet.medium_sent,
                }
            )

        onboard[contact.sat_id] = [packet for packet in sat_queue if packet.packet_id not in selected_ids]

    return SchedulerResult(
        scheduler=scheduler,
        deliveries=deliveries,
        stale_removed=stale_removed,
        expired_removed=expired_removed,
        total_capacity_kb=total_capacity_kb,
    )


# -----------------------------------------------------------------------------
# Metrics and reporting
# -----------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def pct_improvement(new: float, old: float) -> float:
    if old == 0 or math.isnan(old):
        return float("nan")
    return 100.0 * (new - old) / old


def delivery_dataframe(result: SchedulerResult) -> pd.DataFrame:
    if result.deliveries:
        return pd.DataFrame(result.deliveries)
    return pd.DataFrame(
        columns=[
            "scheduler",
            "contact_id",
            "sat_id",
            "packet_id",
            "packet_class",
            "product_class",
            "parcel_id",
            "created_at_h",
            "delivered_at_h",
            "latency_h",
            "size_kb",
            "utility",
            "raw_request_completed",
            "full_raw_sent",
            "preview_sent",
            "medium_sent",
        ]
    )


def summarize_result(result: SchedulerResult, packets: Sequence[Packet]) -> Dict[str, float]:
    df = delivery_dataframe(result)

    generated_emergency = sum(1 for p in packets if p.packet_class == EMERGENCY)
    generated_regular = sum(1 for p in packets if p.packet_class == REGULAR)
    generated_raw = sum(1 for p in packets if p.packet_class == RAW)

    total_utility = float(df["utility"].sum()) if not df.empty else 0.0
    delivered_kb = float(df["size_kb"].sum()) if not df.empty else 0.0

    if not df.empty:
        emergency_df = df[df["packet_class"] == EMERGENCY]
        regular_df = df[df["packet_class"] == REGULAR]
        raw_df = df[df["packet_class"] == RAW]
    else:
        emergency_df = regular_df = raw_df = pd.DataFrame()

    return {
        "scheduler": result.scheduler,
        "total_utility": total_utility,
        "delivered_kb": delivered_kb,
        "value_per_kb": safe_div(total_utility, delivered_kb),
        "capacity_utilization": safe_div(delivered_kb, result.total_capacity_kb),
        "emergency_delivery_rate": safe_div(len(emergency_df), generated_emergency),
        "regular_fault_delivery_rate": safe_div(len(regular_df), generated_regular),
        "median_emergency_latency_h": float(emergency_df["latency_h"].median()) if not emergency_df.empty else float("nan"),
        "median_regular_latency_h": float(regular_df["latency_h"].median()) if not regular_df.empty else float("nan"),
        "raw_request_completion_rate": safe_div(len(raw_df), generated_raw),
        "full_raw_completion_rate": safe_div(len(raw_df), generated_raw),
        "stale_superseded_removed": float(result.stale_removed),
        "expired_removed": float(result.expired_removed),
        "total_capacity_kb": float(result.total_capacity_kb),
    }


def run_all_schedulers(seed: int, cfg: SimConfig) -> Tuple[List[Packet], List[Contact], Dict[str, SchedulerResult], pd.DataFrame]:
    packets, contacts = build_scenario(seed, cfg)
    results = {scheduler: run_scheduler(scheduler, packets, contacts, cfg) for scheduler in SCHEDULERS}
    summary = pd.DataFrame([summarize_result(results[scheduler], packets) for scheduler in SCHEDULERS])
    return packets, contacts, results, summary


def class_breakdown(results: Dict[str, SchedulerResult]) -> pd.DataFrame:
    rows = []
    for scheduler, result in results.items():
        df = delivery_dataframe(result)
        if df.empty:
            continue
        grouped = df.groupby("product_class", as_index=False).agg(
            delivered_kb=("size_kb", "sum"),
            delivered_utility=("utility", "sum"),
            delivered_count=("packet_id", "count"),
        )
        grouped.insert(0, "scheduler", scheduler)
        rows.append(grouped)

    if not rows:
        return pd.DataFrame(columns=["scheduler", "product_class", "delivered_count", "delivered_kb", "delivered_utility"])
    return pd.concat(rows, ignore_index=True)[
        ["scheduler", "product_class", "delivered_count", "delivered_kb", "delivered_utility"]
    ]


def improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    by_scheduler = summary.set_index("scheduler")
    adaptive = by_scheduler.loc["adaptive"]
    rows = []
    for baseline in ("naive_fifo", "priority_fifo"):
        base = by_scheduler.loc[baseline]
        rows.append(
            {
                "comparison": f"adaptive_vs_{baseline}",
                "total_utility_improvement_pct": pct_improvement(adaptive["total_utility"], base["total_utility"]),
                "value_per_kb_improvement_pct": pct_improvement(adaptive["value_per_kb"], base["value_per_kb"]),
                "capacity_utilization_delta_pct_points": 100.0 * (adaptive["capacity_utilization"] - base["capacity_utilization"]),
                "raw_completion_delta_pct_points": 100.0 * (adaptive["raw_request_completion_rate"] - base["raw_request_completion_rate"]),
                "emergency_delivery_delta_pct_points": 100.0 * (adaptive["emergency_delivery_rate"] - base["emergency_delivery_rate"]),
            }
        )
    return pd.DataFrame(rows)


def monte_carlo_medians(seeds: Iterable[int], cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames: List[pd.DataFrame] = []
    improvement_frames: List[pd.DataFrame] = []

    for seed in seeds:
        _, _, _, summary = run_all_schedulers(seed, cfg)
        summary = summary.copy()
        summary["seed"] = seed
        summary_frames.append(summary)

        improvements = improvement_table(summary)
        improvements["seed"] = seed
        improvement_frames.append(improvements)

    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_improvements = pd.concat(improvement_frames, ignore_index=True)

    scheduler_medians = all_summary.groupby("scheduler", as_index=False).agg(
        median_emergency_delivery_rate=("emergency_delivery_rate", "median"),
        median_full_raw_completion_rate=("full_raw_completion_rate", "median"),
        median_capacity_utilization=("capacity_utilization", "median"),
        median_total_utility=("total_utility", "median"),
        median_value_per_kb=("value_per_kb", "median"),
    )

    improvement_medians = all_improvements.groupby("comparison", as_index=False).agg(
        median_total_utility_improvement_pct=("total_utility_improvement_pct", "median"),
        median_value_per_kb_improvement_pct=("value_per_kb_improvement_pct", "median"),
        median_capacity_utilization_delta_pct_points=("capacity_utilization_delta_pct_points", "median"),
        median_raw_completion_delta_pct_points=("raw_completion_delta_pct_points", "median"),
        median_emergency_delivery_delta_pct_points=("emergency_delivery_delta_pct_points", "median"),
    )

    return scheduler_medians, improvement_medians


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scheduler",
        "total_utility",
        "delivered_kb",
        "value_per_kb",
        "capacity_utilization",
        "emergency_delivery_rate",
        "regular_fault_delivery_rate",
        "median_emergency_latency_h",
        "median_regular_latency_h",
        "raw_request_completion_rate",
        "stale_superseded_removed",
    ]
    return summary[cols].copy()


def print_report(single_seed: int = 7, mc_seed_count: int = 30) -> None:
    cfg = SimConfig()
    packets, contacts, results, summary = run_all_schedulers(single_seed, cfg)

    print("CoverGuard downlink scheduler simulation")
    print("=" * 48)
    print(f"Single scenario seed: {single_seed}")
    print(f"Simulation hours: {cfg.sim_hours:.0f}")
    print(f"Satellites: {cfg.n_satellites}")
    print(f"Generated packets: {len(packets)}")
    print(f"Generated emergency faults: {sum(p.packet_class == EMERGENCY for p in packets)}")
    print(f"Generated regular faults: {sum(p.packet_class == REGULAR for p in packets)}")
    print(f"Generated full raw-data requests: {sum(p.packet_class == RAW for p in packets)}")
    print(f"Ground contacts: {len(contacts)}")
    print(f"Total available capacity per scheduler: {sum(c.capacity_kb for c in contacts):,} KB")
    print()

    print("Single-seed scheduler summary")
    print(format_summary_table(summary).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()

    print("Adaptive improvements")
    print(improvement_table(summary).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()

    breakdown = class_breakdown(results)
    print("Delivered KB and value by packet/product class")
    print(breakdown.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()

    # Raw completion called out separately because full raw completion is not interchangeable
    # with fault utility or partial raw review products.
    raw_rates = summary.loc[
        summary["scheduler"].isin(["priority_fifo", "adaptive"]),
        ["scheduler", "raw_request_completion_rate", "full_raw_completion_rate"],
    ]
    print("Full raw request completion: adaptive vs priority FIFO")
    print(raw_rates.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()

    seeds = range(1000, 1000 + mc_seed_count)
    scheduler_medians, improvement_medians = monte_carlo_medians(seeds, cfg)

    print(f"Monte Carlo medians across {mc_seed_count} seeds")
    print("Scheduler medians")
    print(scheduler_medians.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print()
    print("Adaptive improvement medians")
    print(improvement_medians.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))


if __name__ == "__main__":
    print_report(single_seed=7, mc_seed_count=30)
