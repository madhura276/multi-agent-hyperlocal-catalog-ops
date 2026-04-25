from __future__ import annotations

from models import (
    Category,
    Difficulty,
    ExpectedRecordOutcome,
    InventoryRecord,
    RecordStatus,
    TaskDefinition,
    Unit,
)


TASKS = {
    "easy_single_store_cleanup": TaskDefinition(
        task_id="easy_single_store_cleanup",
        title="Single-store cleanup with low ambiguity",
        difficulty=Difficulty.EASY,
        objective=(
            "Coordinate between curation and oversight agents to clean obvious "
            "inventory issues in a single-store batch."
        ),
        records=[
            InventoryRecord(
                record_id="easy_1",
                store_id="store_alpha",
                raw_title="Coca cola 1000ml",
                price=48.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="easy_2",
                store_id="store_alpha",
                raw_title="Amul taaza toned milk 500 ML",
                price=28.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="easy_3",
                store_id="store_alpha",
                raw_title="Banana 6 pc",
                price=35.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="easy_4",
                store_id="store_alpha",
                raw_title="Aashirvaad atta 5kg",
                price=289.0,
                source="merchant_upload",
            ),
        ],
        policy_snippets=[
            "Normalize obvious product titles into clean retail-style names.",
            "Normalize sizes using only allowed units: g, kg, ml, l, pcs.",
            "Assign categories only when the product identity is clear.",
            "For low-ambiguity records, prefer approval over escalation.",
        ],
        hidden_signals={
            "curation_agent": {
                "title_hints": {
                    "easy_1": "Coca Cola 1 L",
                    "easy_2": "Amul Taaza Toned Milk 500 Ml",
                    "easy_3": "Banana 6 Pcs",
                    "easy_4": "Aashirvaad Atta 5 Kg",
                }
            },
            "pricing_agent": {
                "price_reference": {
                    "easy_1": {"expected_band": [40.0, 60.0]},
                    "easy_2": {"expected_band": [20.0, 40.0]},
                    "easy_3": {"expected_band": [20.0, 50.0]},
                    "easy_4": {"expected_band": [240.0, 330.0]},
                }
            },
        },
        expected_outcomes=[
            ExpectedRecordOutcome(
                record_id="easy_1",
                normalized_title="Coca Cola 1 L",
                category=Category.BEVERAGES,
                quantity_value=1.0,
                quantity_unit=Unit.L,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="easy_2",
                normalized_title="Amul Taaza Toned Milk 500 Ml",
                category=Category.DAIRY,
                quantity_value=500.0,
                quantity_unit=Unit.ML,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="easy_3",
                normalized_title="Banana 6 Pcs",
                category=Category.PRODUCE,
                pack_count=6,
                quantity_unit=Unit.PCS,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="easy_4",
                normalized_title="Aashirvaad Atta 5 Kg",
                category=Category.STAPLES,
                quantity_value=5.0,
                quantity_unit=Unit.KG,
                status=RecordStatus.CLEANED,
            ),
        ],
        max_steps=16,
    ),
    "medium_multistore_conflict": TaskDefinition(
        task_id="medium_multistore_conflict",
        title="Multi-store duplicate and pricing conflict",
        difficulty=Difficulty.MEDIUM,
        objective=(
            "Coordinate curation, deduplication, pricing, and oversight agents to "
            "resolve duplicates and pricing conflicts safely."
        ),
        records=[
            InventoryRecord(
                record_id="med_1",
                store_id="store_beta",
                raw_title="Coke 1 ltr",
                price=48.0,
                source="pos_export",
            ),
            InventoryRecord(
                record_id="med_2",
                store_id="store_beta",
                raw_title="Coca Cola 1000 ml",
                price=49.0,
                source="catalog_sync",
            ),
            InventoryRecord(
                record_id="med_3",
                store_id="store_beta",
                raw_title="Surf exel easy wash 1kg",
                price=999.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="med_4",
                store_id="store_gamma",
                raw_title="Amul Taaza Milk 500ml pack",
                price=29.0,
                source="catalog_sync",
            ),
            InventoryRecord(
                record_id="med_5",
                store_id="store_gamma",
                raw_title="Bananas pack of 6",
                price=38.0,
                source="merchant_upload",
            ),
        ],
        policy_snippets=[
            "Only merge records that clearly refer to the same product in the same store.",
            "Price corrections should only be applied when the anomaly is obvious.",
            "Escalate if duplicate resolution is uncertain.",
            "Oversight should reject unsafe merges and approve clear corrections.",
        ],
        hidden_signals={
            "dedupe_agent": {
                "duplicate_candidates": [["med_1", "med_2"]],
            },
            "pricing_agent": {
                "price_reference": {
                    "med_3": {"expected_band": [250.0, 400.0]},
                }
            },
            "oversight_agent": {
                "risk_hints": {
                    "med_1_med_2": "same_store_clear_duplicate",
                    "med_3": "obvious_price_outlier",
                }
            },
        },
        expected_outcomes=[
            ExpectedRecordOutcome(
                record_id="med_1",
                normalized_title="Coca Cola 1 L",
                category=Category.BEVERAGES,
                quantity_value=1.0,
                quantity_unit=Unit.L,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="med_2",
                merged_into="med_1",
                status=RecordStatus.MERGED,
            ),
            ExpectedRecordOutcome(
                record_id="med_3",
                normalized_title="Surf Excel Easy Wash 1 Kg",
                category=Category.HOUSEHOLD,
                quantity_value=1.0,
                quantity_unit=Unit.KG,
                price=320.0,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="med_4",
                normalized_title="Amul Taaza Milk 500 Ml Pack",
                category=Category.DAIRY,
                quantity_value=500.0,
                quantity_unit=Unit.ML,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="med_5",
                normalized_title="Bananas Pack Of 6",
                category=Category.PRODUCE,
                pack_count=6,
                quantity_unit=Unit.PCS,
                status=RecordStatus.CLEANED,
            ),
        ],
        max_steps=20,
    ),
    "hard_ambiguous_oversight_batch": TaskDefinition(
        task_id="hard_ambiguous_oversight_batch",
        title="Ambiguous multi-source batch requiring oversight",
        difficulty=Difficulty.HARD,
        objective=(
            "Handle ambiguous near-duplicates and risky corrections safely, with "
            "oversight preferring escalation when evidence is incomplete."
        ),
        records=[
            InventoryRecord(
                record_id="hard_1",
                store_id="store_delta",
                raw_title="Coke Zero can 300ml",
                price=38.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="hard_2",
                store_id="store_delta",
                raw_title="Coca Cola Zero 330 ml can",
                price=40.0,
                source="catalog_sync",
            ),
            InventoryRecord(
                record_id="hard_3",
                store_id="store_delta",
                raw_title="Tomato loose 1kg",
                price=32.0,
                source="pos_export",
            ),
            InventoryRecord(
                record_id="hard_4",
                store_id="store_delta",
                raw_title="Tomatoes 1 kg pack",
                price=36.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="hard_5",
                store_id="store_delta",
                raw_title="Ariel matic front load 2kg",
                price=999.0,
                source="merchant_upload",
            ),
            InventoryRecord(
                record_id="hard_6",
                store_id="store_delta",
                raw_title="Safal frozen green peas 500 g",
                price=72.0,
                source="catalog_sync",
            ),
            InventoryRecord(
                record_id="hard_7",
                store_id="store_delta",
                raw_title="Green peas 500 g",
                price=44.0,
                source="merchant_upload",
            ),
        ],
        policy_snippets=[
            "Do not merge near-duplicates when packaging or product identity differs.",
            "Packaged and loose produce should usually be escalated instead of merged.",
            "A pricing anomaly may be corrected only if the expected range is clear.",
            "Safety is preferred over aggressive automation in hard tasks.",
        ],
        hidden_signals={
            "dedupe_agent": {
                "duplicate_candidates": [
                    ["hard_1", "hard_2"],
                    ["hard_3", "hard_4"],
                    ["hard_6", "hard_7"],
                ]
            },
            "pricing_agent": {
                "price_reference": {
                    "hard_5": {"expected_band": [250.0, 420.0]},
                }
            },
            "oversight_agent": {
                "risk_hints": {
                    "hard_1_hard_2": "same_family_but_size_diff",
                    "hard_3_hard_4": "loose_vs_packaged_produce",
                    "hard_6_hard_7": "frozen_vs_fresh_or_unspecified",
                    "hard_5": "obvious_price_outlier",
                }
            },
        },
        expected_outcomes=[
            ExpectedRecordOutcome(
                record_id="hard_1",
                normalized_title="Coke Zero Can 300 Ml",
                category=Category.BEVERAGES,
                quantity_value=300.0,
                quantity_unit=Unit.ML,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="hard_2",
                normalized_title="Coca Cola Zero 330 Ml Can",
                category=Category.BEVERAGES,
                quantity_value=330.0,
                quantity_unit=Unit.ML,
                should_flag=True,
                status=RecordStatus.FLAGGED,
            ),
            ExpectedRecordOutcome(
                record_id="hard_3",
                normalized_title="Tomato Loose 1 Kg",
                category=Category.PRODUCE,
                quantity_value=1.0,
                quantity_unit=Unit.KG,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="hard_4",
                normalized_title="Tomatoes 1 Kg Pack",
                category=Category.PRODUCE,
                quantity_value=1.0,
                quantity_unit=Unit.KG,
                should_flag=True,
                status=RecordStatus.FLAGGED,
            ),
            ExpectedRecordOutcome(
                record_id="hard_5",
                normalized_title="Ariel Matic Front Load 2 Kg",
                category=Category.HOUSEHOLD,
                quantity_value=2.0,
                quantity_unit=Unit.KG,
                price=320.0,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="hard_6",
                normalized_title="Safal Frozen Green Peas 500 G",
                category=Category.FROZEN,
                quantity_value=500.0,
                quantity_unit=Unit.G,
                status=RecordStatus.CLEANED,
            ),
            ExpectedRecordOutcome(
                record_id="hard_7",
                normalized_title="Green Peas 500 G",
                category=Category.PRODUCE,
                quantity_value=500.0,
                quantity_unit=Unit.G,
                should_flag=True,
                status=RecordStatus.FLAGGED,
            ),
        ],
        max_steps=24,
    ),
}

TASK_ORDER = [
    "easy_single_store_cleanup",
    "medium_multistore_conflict",
    "hard_ambiguous_oversight_batch",
]

DEFAULT_TASK_ID = TASK_ORDER[0]
