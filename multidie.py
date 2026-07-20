"""
Story 1c: Multi-die product traceability (Case B of the 2x2 matrix).

Case A (single die at CP -> one product at FT) is the default behaviour of
the rest of Story 1 (see assembly.py). Case B packages MULTIPLE known-good
die (each from its own CP population / die size) into ONE FT product
("chiplet" / multi-chip module / SiP).

Per spec 1.c.ii, simple config = 3 different die, each a different size:

    role   | example          | die size (mm) | nominal CP yield
    -------|------------------|----------------|------------------
    logic  | compute chiplet  | 10.0 x 10.0    | 85%
    memory | memory stack     |  5.0 x  5.0    | 90%
    rf     | RF front-end     |  3.0 x  3.0    | 92%

  B.1 "good case"     (`full_trace`)    -- every component keeps a valid,
                       traceable ECID.
  B.2 "annoying case" (`partial_trace`) -- 2 of 3 components are traceable;
                       the 3rd role never burns an ECID at all (that die's
                       process doesn't support it), so its contribution to
                       the product is permanently untraceable back to CP.

A product FAILS FT if ANY of its components fails (typical MCM behaviour).
Traceability is independent of pass/fail: a product can pass FT but still be
only PARTIALLY traceable back to CP -- a real quality risk if that
untraceable component turns out bad in the field.

This is an intentional simplification of full wafer-map realism for the
component populations (per the spec's own "for demo purposes we can set a
simple case" guidance) -- each component role is an independent Bernoulli
known-good-die pool, not a full spatial signature run.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from ecid import format_ecid, ECID_MODE_PLAIN

COMPONENT_ROLES: Tuple[dict, ...] = (
    {"role": "logic",  "die_width": 10.0, "die_height": 10.0, "cp_yield": 0.85},
    {"role": "memory", "die_width": 5.0,  "die_height": 5.0,  "cp_yield": 0.90},
    {"role": "rf",     "die_width": 3.0,  "die_height": 3.0,  "cp_yield": 0.92},
)
COMPONENT_ROLE_NAMES: Tuple[str, ...] = tuple(spec["role"] for spec in COMPONENT_ROLES)

MULTIDIE_MODE_FULL = "full_trace"       # B.1
MULTIDIE_MODE_PARTIAL = "partial_trace"  # B.2
MULTIDIE_MODES = (MULTIDIE_MODE_FULL, MULTIDIE_MODE_PARTIAL)


@dataclass
class ComponentPick:
    role: str
    cp_lot_id: str
    cp_die_index: int
    ecid: str          # "" when this role has no traceability (B.2)
    cp_pass: bool
    ft_pass: bool = True


@dataclass
class MultiDieProduct:
    product_id: str
    ft_lot_id: str
    components: List[ComponentPick] = field(default_factory=list)

    @property
    def ft_pass(self) -> bool:
        return all(c.ft_pass for c in self.components)

    @property
    def fully_traceable(self) -> bool:
        return all(bool(c.ecid) for c in self.components)


def component_cp_lot_id(base_lot_id: str, role: str) -> str:
    return f"{base_lot_id}_{role.upper()}"


def generate_component_pool(
    role: str,
    base_lot_id: str,
    n_needed: int,
    cp_yield: float,
    traceable: bool,
    seed: int,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> List[ComponentPick]:
    """CP-side pool of known-good die for one component role.

    Draws candidates until `n_needed` PASSING die are collected. Each
    candidate is an independent Bernoulli(cp_yield) draw (see module
    docstring for why full spatial signatures are skipped here).
    """
    rng = random.Random(seed)
    lot_id = component_cp_lot_id(base_lot_id, role)
    passers: List[ComponentPick] = []
    idx = 0
    max_attempts = max(1000, n_needed * 50)
    attempts = 0
    while len(passers) < n_needed and attempts < max_attempts:
        idx += 1
        attempts += 1
        if rng.random() >= cp_yield:
            continue
        ecid = format_ecid(lot_id, 1, idx, 0, mode=ecid_mode) if traceable else ""
        passers.append(ComponentPick(
            role=role, cp_lot_id=lot_id, cp_die_index=idx,
            ecid=ecid, cp_pass=True,
        ))
    return passers


def assemble_multidie_products(
    base_lot_id: str,
    n_products: int,
    mode: str = MULTIDIE_MODE_FULL,
    untraceable_role: str = "rf",
    baseline_ft_fallout: float = 0.05,
    ft_lot_id: str = "",
    seed: int = 0,
    ecid_mode: str = ECID_MODE_PLAIN,
) -> Tuple[List[MultiDieProduct], dict]:
    """Build `n_products` packaged multi-die products from the 3 component roles."""
    ft_lot_id = ft_lot_id or f"FT_{base_lot_id}_MCM"
    rng = random.Random(seed + 6060)
    manifest: Dict[str, object] = {"warnings": [], "stats": {}}

    pools: Dict[str, List[ComponentPick]] = {}
    for i, spec in enumerate(COMPONENT_ROLES):
        role = spec["role"]
        traceable = not (mode == MULTIDIE_MODE_PARTIAL and role == untraceable_role)
        pools[role] = generate_component_pool(
            role, base_lot_id, n_products, spec["cp_yield"], traceable,
            seed=seed + i * 991, ecid_mode=ecid_mode,
        )
        if len(pools[role]) < n_products:
            manifest["warnings"].append(
                f"Component '{role}': only {len(pools[role])}/{n_products} "
                "known-good die available; some products short-built."
            )

    n_buildable = min((len(p) for p in pools.values()), default=0)
    products: List[MultiDieProduct] = []
    for i in range(n_buildable):
        comps: List[ComponentPick] = []
        for spec in COMPONENT_ROLES:
            base = pools[spec["role"]][i]
            ft_pass = rng.random() >= baseline_ft_fallout
            comps.append(ComponentPick(
                role=base.role, cp_lot_id=base.cp_lot_id,
                cp_die_index=base.cp_die_index, ecid=base.ecid,
                cp_pass=True, ft_pass=ft_pass,
            ))
        products.append(MultiDieProduct(
            product_id=f"{ft_lot_id}_P{i + 1:05d}",
            ft_lot_id=ft_lot_id,
            components=comps,
        ))

    manifest["stats"]["products"] = float(len(products))
    manifest["stats"]["fully_traceable"] = float(
        sum(1 for p in products if p.fully_traceable))
    manifest["stats"]["ft_pass"] = float(sum(1 for p in products if p.ft_pass))
    manifest["stats"]["untraceable_role"] = (
        untraceable_role if mode == MULTIDIE_MODE_PARTIAL else ""
    )
    return products, manifest


def build_multidie_df(products: Sequence[MultiDieProduct]) -> pd.DataFrame:
    """One row per packaged product, with per-component detail columns."""
    rows = []
    for p in products:
        row = {
            "ProductId": p.product_id,
            "FtLot": p.ft_lot_id,
            "ProductFtPass": int(p.ft_pass),
            "FullyTraceable": bool(p.fully_traceable),
        }
        for c in p.components:
            prefix = c.role.capitalize()
            row[f"{prefix}CpLot"] = c.cp_lot_id
            row[f"{prefix}Ecid"] = c.ecid
            row[f"{prefix}FtPass"] = int(c.ft_pass)
        rows.append(row)
    return pd.DataFrame(rows)
