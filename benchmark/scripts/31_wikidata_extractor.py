"""Wikidata entity evolution extractor.

Queries the Wikidata API for entities with attribute changes over time,
extracting revision history to build bitemporal scenario seeds.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "Wikidata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target entities across 6 domains
# Format: (QID, Chinese label hint, domain, key properties to track)
TARGET_ENTITIES: List[Dict[str, Any]] = [
    # === Politics (职位变更) ===
    {"qid": "Q9168", "label": "安倍晋三", "domain": "politics", "props": ["P39"]},  # Abe Shinzo
    {"qid": "Q47558", "label": "默克尔", "domain": "politics", "props": ["P39"]},  # Merkel
    {"qid": "Q5623", "label": "特朗普", "domain": "politics", "props": ["P39"]},  # Trump
    {"qid": "Q6279", "label": "马克龙", "domain": "politics", "props": ["P39"]},  # Macron
    {"qid": "Q180860", "label": "岸田文雄", "domain": "politics", "props": ["P39"]},
    {"qid": "Q7747", "label": "约翰逊", "domain": "politics", "props": ["P39"]},  # Boris Johnson

    # === Tech companies (总部、CEO变更) ===
    {"qid": "Q1299", "label": "苹果公司", "domain": "tech", "props": ["P159", "P169"]},
    {"qid": "Q2283", "label": "微软", "domain": "tech", "props": ["P159", "P169"]},
    {"qid": "Q688", "label": "英特尔", "domain": "tech", "props": ["P159", "P169"]},
    {"qid": "Q9682", "label": "阿里巴巴", "domain": "tech", "props": ["P159", "P169"]},
    {"qid": "Q867418", "label": "字节跳动", "domain": "tech", "props": ["P159", "P169"]},
    {"qid": "Q7503994", "label": "华为", "domain": "tech", "props": ["P159"]},

    # === Sports (队伍、成绩) ===
    {"qid": "Q8682", "label": "皇家马德里", "domain": "sports", "props": ["P527", "P1346"]},
    {"qid": "Q8683", "label": "巴塞罗那", "domain": "sports", "props": ["P1346"]},
    {"qid": "Q483014", "label": "曼联", "domain": "sports", "props": ["P1346"]},
    {"qid": "Q715271", "label": "广州恒大", "domain": "sports", "props": ["P1346"]},

    # === Science (奖项、职位) ===
    {"qid": "Q935", "label": "爱因斯坦", "domain": "science", "props": ["P166"]},
    {"qid": "Q5582", "label": "屠呦呦", "domain": "science", "props": ["P166"]},
    {"qid": "Q7213", "label": "杨振宁", "domain": "science", "props": ["P166"]},

    # === Entertainment (作品、获奖) ===
    {"qid": "Q104098", "label": "宫崎骏", "domain": "entertainment", "props": ["P166"]},
    {"qid": "Q1138588", "label": "李安", "domain": "entertainment", "props": ["P166"]},
    {"qid": "Q61476", "label": "莫言", "domain": "entertainment", "props": ["P166"]},

    # === Geography (人口、面积变更) ===
    {"qid": "Q956", "label": "北京", "domain": "geography", "props": ["P1082"]},
    {"qid": "Q8686", "label": "上海", "domain": "geography", "props": ["P1082"]},
    {"qid": "Q1490", "label": "东京", "domain": "geography", "props": ["P1082"]},
]

PROPERTY_LABELS = {
    "P39": "担任职务",
    "P106": "职业",
    "P159": "总部所在地",
    "P169": "首席执行官",
    "P527": "组成部分",
    "P1346": "获奖",
    "P166": "获得奖项",
    "P1082": "人口",
}

WD_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "BiTempQA-Benchmark/1.0 (academic research)"}


def get_entity_data(qid: str) -> Optional[Dict]:
    """Fetch entity data from Wikidata API."""
    params = {
        "action": "wbgetentities",
        "ids": qid,
        "format": "json",
        "props": "labels|claims",
        "languages": "zh|en",
    }
    try:
        resp = requests.get(WD_API, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json().get("entities", {}).get(qid)
    except Exception as e:
        logger.warning(f"Failed to fetch {qid}: {e}")
        return None


def get_entity_revisions(qid: str, limit: int = 50) -> List[Dict]:
    """Fetch revision history for an entity."""
    params = {
        "action": "query",
        "titles": qid,
        "prop": "revisions",
        "rvprop": "ids|timestamp|size|comment",
        "rvlimit": limit,
        "format": "json",
    }
    try:
        resp = requests.get(WD_API, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            return page_data.get("revisions", [])
    except Exception as e:
        logger.warning(f"Failed to fetch revisions for {qid}: {e}")
    return []


def extract_property_claims(entity_data: Dict, props: List[str]) -> List[Dict]:
    """Extract property claims with qualifiers (time, etc.)."""
    claims = entity_data.get("claims", {})
    results = []

    for prop_id in props:
        if prop_id not in claims:
            continue

        for claim in claims[prop_id]:
            # Get main value
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})

            if isinstance(value, dict) and "id" in value:
                # Wikidata entity reference
                target_qid = value["id"]
                target_label = value.get("text", target_qid)
            elif isinstance(value, dict) and "amount" in value:
                target_label = value["amount"]
            elif isinstance(value, str):
                target_label = value
            else:
                continue

            # Get time qualifiers
            qualifiers = claim.get("qualifiers", {})
            start_time = None
            end_time = None

            for qual_prop in ["P580", "P582"]:  # start time, end time
                if qual_prop in qualifiers:
                    for qual in qualifiers[qual_prop]:
                        qval = qual.get("datavalue", {}).get("value", {})
                        if isinstance(qval, dict) and "time" in qval:
                            t = qval["time"]
                            if qual_prop == "P580":
                                start_time = t
                            else:
                                end_time = t

            # Also check rank (preferred > normal > deprecated)
            rank = claim.get("rank", "normal")

            results.append({
                "property": prop_id,
                "property_label": PROPERTY_LABELS.get(prop_id, prop_id),
                "value": target_label,
                "value_qid": value.get("id") if isinstance(value, dict) else None,
                "start_time": start_time,
                "end_time": end_time,
                "rank": rank,
            })

    return results


def get_chinese_label(entity_data: Dict) -> str:
    """Get Chinese label for entity."""
    labels = entity_data.get("labels", {})
    if "zh" in labels:
        return labels["zh"]["value"]
    if "zh-hans" in labels:
        return labels["zh-hans"]["value"]
    if "en" in labels:
        return labels["en"]["value"]
    return entity_data.get("id", "Unknown")


def resolve_qid_label(qid: str) -> str:
    """Resolve a QID to its Chinese label."""
    data = get_entity_data(qid)
    if data:
        return get_chinese_label(data)
    return qid


def entity_to_scenarios(
    entity_meta: Dict,
    entity_data: Dict,
    claims: List[Dict],
    revisions: List[Dict],
) -> List[Dict]:
    """Convert entity data into BiTempQA scenario seeds."""
    qid = entity_meta["qid"]
    label = get_chinese_label(entity_data)
    domain = entity_meta["domain"]

    scenarios = []

    # Group claims by property
    from collections import defaultdict
    by_prop = defaultdict(list)
    for c in claims:
        by_prop[c["property"]].append(c)

    for prop_id, prop_claims in by_prop.items():
        # Only use properties with temporal changes (multiple claims or time qualifiers)
        if len(prop_claims) < 2:
            # Check if it has time qualifiers (even single claim is useful)
            has_time = any(c["start_time"] or c["end_time"] for c in prop_claims)
            if not has_time:
                continue

        # Build memory writes from claim history
        writes = []
        for claim in prop_claims:
            value_label = claim["value"]
            if claim["value_qid"] and len(claim["value_qid"]) < 15:
                # Resolve to Chinese label (with caching)
                try:
                    value_label = resolve_qid_label(claim["value_qid"])
                    time.sleep(0.5)  # Rate limit
                except:
                    pass

            start = claim.get("start_time")
            end = claim.get("end_time")

            if start:
                event_time = start.rstrip("Z").replace("+00:00", "")[:10]
                # Use revision timestamp as record_time (when the claim was added to Wikidata)
                record_time = None
                for rev in revisions:
                    ts = rev.get("timestamp", "")
                    if ts:
                        rev_date = ts[:10]
                        if rev_date >= event_time:
                            record_time = rev_date
                            break

                prop_label = claim["property_label"]
                text = f"{label}的{prop_label}变更为{value_label}。"

                writes.append({
                    "text": text,
                    "event_time": event_time,
                    "record_time": record_time or event_time,
                    "source": "wikidata",
                })

        if len(writes) >= 2:
            scenarios.append({
                "entity_qid": qid,
                "entity_label": label,
                "domain": domain,
                "property": prop_id,
                "property_label": PROPERTY_LABELS.get(prop_id, prop_id),
                "writes": writes,
                "num_changes": len(writes),
            })

    return scenarios


def main():
    logger.info(f"Extracting data for {len(TARGET_ENTITIES)} Wikidata entities")

    all_scenarios = []
    failed = []

    for i, entity_meta in enumerate(TARGET_ENTITIES):
        qid = entity_meta["qid"]
        logger.info(f"[{i+1}/{len(TARGET_ENTITIES)}] Processing {qid} ({entity_meta['label']})...")

        # Fetch entity data
        entity_data = get_entity_data(qid)
        if not entity_data:
            failed.append(qid)
            continue

        # Fetch revision history
        revisions = get_entity_revisions(qid, limit=50)

        # Extract claims
        claims = extract_property_claims(entity_data, entity_meta["props"])
        if not claims:
            logger.info(f"  No relevant claims found for {qid}")
            continue

        # Convert to scenarios
        scenarios = entity_to_scenarios(entity_meta, entity_data, claims, revisions)
        logger.info(f"  Found {len(scenarios)} scenario groups, {sum(len(s['writes']) for s in scenarios)} writes")
        all_scenarios.extend(scenarios)

        time.sleep(1)  # Rate limiting

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total scenario groups: {len(all_scenarios)}")
    logger.info(f"Total writes: {sum(len(s['writes']) for s in all_scenarios)}")
    logger.info(f"Failed entities: {failed}")

    # Domain distribution
    from collections import Counter
    domain_dist = Counter(s["domain"] for s in all_scenarios)
    logger.info(f"Domain distribution: {dict(domain_dist)}")

    # Save
    output_file = OUTPUT_DIR / "wikidata_scenarios.json"
    with open(output_file, "w") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
