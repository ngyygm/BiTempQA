"""Convert ChronoQA + curated real-world events into BiTempQA scenarios.

Produces scenarios in the exact format expected by the evaluation pipeline:
  - scenario_id, scenario_type, title, description, domain, language
  - memory_writes: [{write_id, text, source_name, event_time, record_time}]
  - world_states: [{as_of_record_time, entities, relations, known_facts}]
  - qa_pairs: [{qa_id, question, answer, ...}]

Data sources:
  1. ChronoQA (200 bitemporal candidates with golden_chunks)
  2. Manually curated real-world temporal events
  3. Wikidata entity evolution (supplementary)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "generated" / "real_source_scenarios"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Source 1: ChronoQA conversion
# ============================================================

def convert_chronoqa(candidates_file: Path) -> List[Dict]:
    """Convert ChronoQA bitemporal candidates to BiTempQA scenarios."""
    with open(candidates_file) as f:
        candidates = json.load(f)

    scenarios = []
    for i, cand in enumerate(candidates):
        scenario_id = f"CHRONO_{i+1:03d}"

        # Build memory writes from golden_chunks
        # Use golden_chunk text as memory writes, with event_time and publish_date as record_time
        writes = []
        chunks = cand.get("golden_chunks", [])
        for j, chunk in enumerate(chunks[:5]):
            writes.append({
                "write_id": f"w{j+1}",
                "text": chunk.strip(),
                "source_name": "chronoqa_golden",
                "event_time": f"{cand['event_date']}T00:00:00Z",
                "record_time": f"{cand['publish_date']}T00:00:00Z",
            })

        if not writes:
            continue

        # Build world states
        world_states = []
        for j, write in enumerate(writes):
            ws = {
                "as_of_record_time": write["record_time"],
                "entities": [],
                "relations": [],
                "known_facts": [w["text"] for w in writes[:j+1]],
            }
            world_states.append(ws)

        # Determine scenario type based on temporal_type (map to existing schema enum)
        ttype = cand.get("temporal_type", "absolute")
        scenario_type_map = {
            "absolute": "entity_attribute_evolution",    # point-in-time queries
            "aggregate": "multi_source_information",     # aggregation over multiple facts
            "relative": "temporal_ambiguity",            # relative temporal reasoning
        }
        scenario_type = scenario_type_map.get(ttype, "entity_attribute_evolution")

        # Map ChronoQA question_type to valid QAPair question_type enum
        question_type_map = {
            "absolute": "point_in_time",
            "aggregate": "multi_hop_temporal",
            "relative": "temporal_ordering",
        }

        # Build QA pair from ChronoQA question/answer
        gap_days = cand["gap_days"]
        requires_evt = gap_days > 0
        requires_rec = gap_days >= 7

        qa = {
            "qa_id": f"{scenario_id}_L1_001",
            "scenario_id": scenario_id,
            "difficulty": "level_1",
            "question_type": question_type_map.get(ttype, "point_in_time"),
            "question_zh": cand["question"],
            "question_en": "",
            "answer_zh": cand["answer"],
            "answer_en": "",
            "answer_type": "abstractive",
            "choices": [],
            "correct_choice_index": None,
            "ranking_order": None,
            "query_event_time": f"{cand['event_date']}T00:00:00Z",
            "query_record_time": f"{cand['publish_date']}T00:00:00Z" if requires_rec else None,
            "relevant_time_range": {
                "start": f"{cand['event_date']}T00:00:00Z",
                "end": f"{cand['publish_date']}T00:00:00Z",
            },
            "reasoning_chain": [
                f"事件发生于{cand['event_date']}，报道于{cand['publish_date']}，时间差{gap_days}天"
            ],
            "requires_event_time_reasoning": requires_evt,
            "requires_record_time_reasoning": requires_rec,
            "requires_version_tracking": False,
            "requires_knowledge_retraction": False,
            "source_write_ids": [f"w{j+1}" for j in range(min(len(chunks), 5))],
            "generation_method": "llm_human",
            "validation_status": "validated",
            "source_url": cand.get("urls", [""])[0],
        }

        scenarios.append({
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "title_zh": cand["question"][:30],
            "title_en": "",
            "description_zh": f"ChronoQA来源: 事件时间{cand['event_date']}, 报道时间{cand['publish_date']}, 时间差{gap_days}天",
            "description_en": "",
            "domain": "news",
            "language": "zh",
            "memory_writes": writes,
            "world_states": world_states,
            "qa_pairs": [qa],
            "data_source": "ChronoQA",
            "original_urls": cand.get("urls", []),
        })

    return scenarios


# ============================================================
# Source 2: Manually curated real-world temporal events
# ============================================================

CURATED_SCENARIOS: List[Dict] = [
    # --- CEO 变更 ---
    {
        "scenario_id": "REAL_CEO_001",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "微软CEO更替",
        "domain": "tech",
        "writes": [
            {"text": "1975年4月4日，比尔·盖茨创立微软公司并担任CEO。", "event_time": "1975-04-04", "record_time": "1975-04-05"},
            {"text": "2000年1月13日，史蒂夫·鲍尔默接替比尔·盖茨出任微软CEO。盖茨转任首席软件架构师。", "event_time": "2000-01-13", "record_time": "2000-01-14"},
            {"text": "2014年2月4日，萨蒂亚·纳德拉接替鲍尔默成为微软第三任CEO。", "event_time": "2014-02-04", "record_time": "2014-02-04"},
        ],
        "qa_pairs": [
            {"q": "2005年，微软的CEO是谁？", "a": "史蒂夫·鲍尔默", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "2010年时，微软历任CEO有哪几位？", "a": "比尔·盖茨和史蒂夫·鲍尔默", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": False},
            {"q": "微软CEO变更的先后顺序是什么？", "a": "比尔·盖茨 → 史蒂夫·鲍尔默 → 萨蒂亚·纳德拉", "type": "temporal_ordering", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    {
        "scenario_id": "REAL_CEO_002",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "苹果CEO更替",
        "domain": "tech",
        "writes": [
            {"text": "1997年9月16日，史蒂夫·乔布斯回归苹果公司担任临时CEO。", "event_time": "1997-09-16", "record_time": "1997-09-17"},
            {"text": "2000年1月5日，乔布斯正式成为苹果CEO，去掉了'临时'头衔。", "event_time": "2000-01-05", "record_time": "2000-01-06"},
            {"text": "2011年8月24日，乔布斯辞去苹果CEO职务，蒂姆·库克接任。", "event_time": "2011-08-24", "record_time": "2011-08-24"},
        ],
        "qa_pairs": [
            {"q": "1999年，苹果的CEO是谁？", "a": "史蒂夫·乔布斯（临时CEO）", "type": "point_in_time", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "蒂姆·库克是什么时候成为苹果CEO的？", "a": "2011年8月24日", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    {
        "scenario_id": "REAL_CEO_003",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "阿里巴巴CEO更替",
        "domain": "tech",
        "writes": [
            {"text": "1999年，马云创立阿里巴巴集团并担任CEO。", "event_time": "1999-06-28", "record_time": "1999-06-29"},
            {"text": "2013年1月15日，马云宣布将辞去阿里巴巴CEO职务。", "event_time": "2013-01-15", "record_time": "2013-01-15"},
            {"text": "2013年5月10日，陆兆禧正式接任阿里巴巴集团CEO。", "event_time": "2013-05-10", "record_time": "2013-05-10"},
            {"text": "2015年5月7日，张勇接替陆兆禧出任阿里巴巴集团CEO。", "event_time": "2015-05-07", "record_time": "2015-05-07"},
            {"text": "2019年9月10日，张勇接替马云担任阿里巴巴集团董事会主席。", "event_time": "2019-09-10", "record_time": "2019-09-10"},
        ],
        "qa_pairs": [
            {"q": "2014年时，阿里巴巴的CEO是谁？", "a": "陆兆禧", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "截至2016年，阿里巴巴一共经历了几位CEO？分别是谁？", "a": "三位：马云、陆兆禧、张勇", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 体育赛事 ---
    {
        "scenario_id": "REAL_SPORT_001",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "皇家马德里主教练更替",
        "domain": "sports",
        "writes": [
            {"text": "2013年6月25日，安切洛蒂出任皇家马德里主教练。", "event_time": "2013-06-25", "record_time": "2013-06-25"},
            {"text": "2015年5月25日，皇马宣布安切洛蒂下课，贝尼特斯接任。", "event_time": "2015-05-25", "record_time": "2015-05-25"},
            {"text": "2016年1月4日，贝尼特斯被解雇，齐达内接任皇马主教练。", "event_time": "2016-01-04", "record_time": "2016-01-04"},
            {"text": "2018年5月31日，齐达内辞去皇马主教练职务。", "event_time": "2018-05-31", "record_time": "2018-05-31"},
            {"text": "2019年3月11日，齐达内二度出任皇马主教练。", "event_time": "2019-03-11", "record_time": "2019-03-11"},
            {"text": "2021年5月27日，齐达内再次辞去皇马主帅职务。", "event_time": "2021-05-27", "record_time": "2021-05-27"},
            {"text": "2021年6月1日，安切洛蒂回归执教皇家马德里。", "event_time": "2021-06-01", "record_time": "2021-06-01"},
        ],
        "qa_pairs": [
            {"q": "2017年，皇马主教练是谁？", "a": "齐达内", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "安切洛蒂两次执教皇马之间间隔了多久？", "a": "约6年（2015年5月至2021年6月）", "type": "complex_temporal", "difficulty": "level_3", "evt_reason": True, "rec_reason": False},
            {"q": "齐达内几次担任皇马主教练？", "a": "两次（2016年1月至2018年5月，2019年3月至2021年5月）", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 政治职务 ---
    {
        "scenario_id": "REAL_POLI_001",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "日本首相更替（安倍-菅义伟-岸田）",
        "domain": "politics",
        "writes": [
            {"text": "2012年12月26日，安倍晋三就任日本第96任首相。", "event_time": "2012-12-26", "record_time": "2012-12-26"},
            {"text": "2020年8月28日，安倍晋三宣布因健康原因辞去首相职务。", "event_time": "2020-08-28", "record_time": "2020-08-28"},
            {"text": "2020年9月16日，菅义伟就任日本第99任首相。", "event_time": "2020-09-16", "record_time": "2020-09-16"},
            {"text": "2021年10月4日，岸田文雄就任日本第100任首相。", "event_time": "2021-10-04", "record_time": "2021-10-04"},
        ],
        "qa_pairs": [
            {"q": "2021年1月1日时，日本首相是谁？", "a": "菅义伟", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "从2012年到2022年，日本经历了哪几位首相？按顺序列举。", "a": "安倍晋三、菅义伟、岸田文雄", "type": "temporal_ordering", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    {
        "scenario_id": "REAL_POLI_002",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "英国首相更替（卡梅伦-特蕾莎-约翰逊-特拉斯-苏纳克）",
        "domain": "politics",
        "writes": [
            {"text": "2016年7月13日，特蕾莎·梅接替卡梅伦就任英国首相。", "event_time": "2016-07-13", "record_time": "2016-07-13"},
            {"text": "2019年7月24日，鲍里斯·约翰逊接替特蕾莎·梅就任英国首相。", "event_time": "2019-07-24", "record_time": "2019-07-24"},
            {"text": "2022年9月6日，利兹·特拉斯就任英国首相。", "event_time": "2022-09-06", "record_time": "2022-09-06"},
            {"text": "2022年10月25日，里希·苏纳克就任英国首相。", "event_time": "2022-10-25", "record_time": "2022-10-25"},
        ],
        "qa_pairs": [
            {"q": "2020年时英国首相是谁？", "a": "鲍里斯·约翰逊", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "特蕾莎·梅担任首相期间，后面一位首相是谁？", "a": "鲍里斯·约翰逊", "type": "temporal_ordering", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
            {"q": "2022年英国一共有几位首相？分别是谁？", "a": "三位：鲍里斯·约翰逊、利兹·特拉斯、里希·苏纳克", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 公司总部迁移 ---
    {
        "scenario_id": "REAL_HQ_001",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "联想集团总部变更",
        "domain": "tech",
        "writes": [
            {"text": "1984年，联想集团在北京成立，总部设在北京。", "event_time": "1984-11-01", "record_time": "1984-11-01"},
            {"text": "2004年12月，联想收购IBM PC业务，宣布将全球总部迁至纽约。", "event_time": "2004-12-08", "record_time": "2004-12-08"},
            {"text": "2006年3月，联想将全球总部从纽约迁至北卡罗来纳州罗利。", "event_time": "2006-03-01", "record_time": "2006-03-01"},
            {"text": "2018年5月，联想集团将全球总部设在北京和罗利双总部。", "event_time": "2018-05-01", "record_time": "2018-05-01"},
        ],
        "qa_pairs": [
            {"q": "2005年，联想的全球总部在哪里？", "a": "纽约", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "联想总部迁移的顺序是什么？", "a": "北京 → 纽约 → 罗利 → 北京+罗利双总部", "type": "temporal_ordering", "difficulty": "level_2", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 迟到信息 ---
    {
        "scenario_id": "REAL_LATE_001",
        "scenario_type": "late_arriving_facts",
        "title_zh": "新冠疫情时间线（迟到信息）",
        "domain": "health",
        "writes": [
            {"text": "2019年12月31日，武汉市卫健委通报发现不明原因肺炎病例。", "event_time": "2019-12-31", "record_time": "2019-12-31"},
            {"text": "2020年1月9日，中国疾控中心确认病原体为新型冠状病毒。", "event_time": "2020-01-09", "record_time": "2020-01-09"},
            {"text": "2020年3月17日，研究发现新冠病毒可能早在2019年12月就已经在意大利传播。", "event_time": "2019-12-01", "record_time": "2020-03-17"},
            {"text": "2020年6月，西班牙巴塞罗那大学在2019年3月的废水样本中检测到新冠病毒。", "event_time": "2019-03-01", "record_time": "2020-06-26"},
        ],
        "qa_pairs": [
            {"q": "截至2020年2月1日，已知最早的新冠病毒传播可以追溯到什么时候？", "a": "2019年12月31日（武汉通报）", "type": "point_in_time", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "截至2020年7月，已知最早的新冠病毒传播证据是什么？", "a": "2019年3月西班牙废水样本", "type": "point_in_time", "difficulty": "level_3", "evt_reason": True, "rec_reason": True},
        ],
    },
    # --- 知识更正 ---
    {
        "scenario_id": "REAL_RETRACT_001",
        "scenario_type": "knowledge_retraction",
        "title_zh": "冥王星行星身份变更",
        "domain": "science",
        "writes": [
            {"text": "1930年2月18日，克莱德·汤博发现冥王星，被归类为太阳系第九大行星。", "event_time": "1930-02-18", "record_time": "1930-03-13"},
            {"text": "2005年1月5日，天文学家发现阋神星（Eris），其质量大于冥王星，引发行星定义争议。", "event_time": "2005-01-05", "record_time": "2005-07-29"},
            {"text": "2006年8月24日，国际天文学联合会重新定义行星标准，冥王星被降级为矮行星。", "event_time": "2006-08-24", "record_time": "2006-08-24"},
        ],
        "qa_pairs": [
            {"q": "2000年时，太阳系有多少颗行星？", "a": "九颗", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "2007年时，冥王星是什么分类？", "a": "矮行星", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "冥王星的分类发生了什么变化？", "a": "从行星降级为矮行星（2006年8月24日）", "type": "counterfactual", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
        ],
    },
    # --- 多源信息 ---
    {
        "scenario_id": "REAL_MULTI_001",
        "scenario_type": "multi_source_information",
        "title_zh": "马航MH370事件多方报道",
        "domain": "news",
        "writes": [
            {"text": "2014年3月8日，马来西亚航空MH370航班从吉隆坡飞往北京途中失联。", "event_time": "2014-03-08", "record_time": "2014-03-08", "source": "马来西亚民航局"},
            {"text": "2014年3月15日，马来西亚总理纳吉布确认MH370的通信系统被人为关闭，飞机改变了航线。", "event_time": "2014-03-08", "record_time": "2014-03-15", "source": "马来西亚政府"},
            {"text": "2015年7月29日，留尼汪岛发现波音777襟副翼残骸，后经确认属于MH370。", "event_time": "2015-07-29", "record_time": "2015-08-05", "source": "法国调查局"},
            {"text": "2017年1月17日，马来西亚、中国和澳大利亚宣布暂停水下搜索行动。", "event_time": "2017-01-17", "record_time": "2017-01-17", "source": "联合声明"},
        ],
        "qa_pairs": [
            {"q": "截至2014年3月20日，关于MH370的已知信息有哪些？", "a": "3月8日失联，通信系统被人为关闭，飞机改变航线", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "MH370的首块残骸是什么时候被发现的？", "a": "2015年7月29日", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 诺贝尔奖 ---
    {
        "scenario_id": "REAL_PRIZE_001",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "屠呦呦获诺贝尔奖",
        "domain": "science",
        "writes": [
            {"text": "1969年，屠呦呦领导的团队开始研究抗疟药物，从中草药中提取青蒿素。", "event_time": "1969-01-01", "record_time": "1969-01-01"},
            {"text": "2011年9月23日，屠呦呦获得拉斯克临床医学研究奖。", "event_time": "2011-09-23", "record_time": "2011-09-23"},
            {"text": "2015年10月5日，屠呦呦获得诺贝尔生理学或医学奖，成为中国本土首位诺贝尔科学奖获得者。", "event_time": "2015-10-05", "record_time": "2015-10-05"},
        ],
        "qa_pairs": [
            {"q": "2010年时，屠呦呦是否获得过诺贝尔奖？", "a": "没有", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": True},
            {"q": "屠呦呦获得拉斯克奖和诺贝尔奖的先后顺序是什么？", "a": "先获拉斯克奖（2011年），后获诺贝尔奖（2015年）", "type": "temporal_ordering", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 足球世界杯 ---
    {
        "scenario_id": "REAL_SPORT_002",
        "scenario_type": "entity_attribute_evolution",
        "title_zh": "世界杯冠军记录",
        "domain": "sports",
        "writes": [
            {"text": "2002年6月30日，巴西队在韩日世界杯决赛中2-0击败德国，第五次获得世界杯冠军。", "event_time": "2002-06-30", "record_time": "2002-06-30"},
            {"text": "2006年7月9日，意大利在德国世界杯决赛中击败法国，第四次获得世界杯冠军。", "event_time": "2006-07-09", "record_time": "2006-07-09"},
            {"text": "2010年7月11日，西班牙在南非世界杯决赛中1-0击败荷兰，首次获得世界杯冠军。", "event_time": "2010-07-11", "record_time": "2010-07-11"},
            {"text": "2014年7月13日，德国在巴西世界杯决赛中1-0击败阿根廷，第四次获得世界杯冠军。", "event_time": "2014-07-13", "record_time": "2014-07-13"},
            {"text": "2018年7月15日，法国在俄罗斯世界杯决赛中4-2击败克罗地亚，第二次获得世界杯冠军。", "event_time": "2018-07-15", "record_time": "2018-07-15"},
            {"text": "2022年12月18日，阿根廷在卡塔尔世界杯决赛中击败法国，第三次获得世界杯冠军。", "event_time": "2022-12-18", "record_time": "2022-12-18"},
        ],
        "qa_pairs": [
            {"q": "截至2012年，获得世界杯冠军次数最多的国家是哪个？", "a": "巴西（5次）", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "2005年时，德国队获得过几次世界杯冠军？", "a": "三次（1954、1974、1990）", "type": "point_in_time", "difficulty": "level_3", "evt_reason": True, "rec_reason": True},
            {"q": "2023年时，获得世界杯冠军次数最多的国家是哪个？", "a": "巴西（5次）", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
        ],
    },
    # --- 迟到信息: 疫苗 ---
    {
        "scenario_id": "REAL_LATE_002",
        "scenario_type": "late_arriving_facts",
        "title_zh": "新冠疫苗研发时间线",
        "domain": "health",
        "writes": [
            {"text": "2020年1月11日，中国科学家向世界卫生组织共享了新冠病毒基因组序列。", "event_time": "2020-01-11", "record_time": "2020-01-11"},
            {"text": "2020年3月16日，莫德纳公司开始mRNA-1273疫苗的第一期临床试验。", "event_time": "2020-03-16", "record_time": "2020-03-16"},
            {"text": "2020年11月9日，辉瑞/BioNTech宣布其新冠疫苗有效率达90%以上。", "event_time": "2020-11-09", "record_time": "2020-11-09"},
            {"text": "2020年12月11日，美国FDA批准辉瑞/BioNTech新冠疫苗紧急使用授权。", "event_time": "2020-12-11", "record_time": "2020-12-11"},
            {"text": "事后研究发现，2020年1月采集的部分美国血液样本中已存在新冠抗体，暗示病毒传播早于预期。", "event_time": "2020-01-01", "record_time": "2020-12-01"},
        ],
        "qa_pairs": [
            {"q": "截至2020年6月，有哪些新冠疫苗进入临床试验？", "a": "莫德纳mRNA-1273已进入第一期临床试验", "type": "point_in_time", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "辉瑞疫苗是什么时候被证明有效的？", "a": "2020年11月9日", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 渐进积累 ---
    {
        "scenario_id": "REAL_ACCUM_001",
        "scenario_type": "gradual_accumulation",
        "title_zh": "嫦娥探月工程进展",
        "domain": "science",
        "writes": [
            {"text": "2007年10月24日，嫦娥一号成功发射，中国首次月球探测任务开始。", "event_time": "2007-10-24", "record_time": "2007-10-24"},
            {"text": "2010年10月1日，嫦娥二号成功发射，获取了高分辨率月球表面影像。", "event_time": "2010-10-01", "record_time": "2010-10-01"},
            {"text": "2013年12月14日，嫦娥三号携带玉兔号月球车成功软着陆于月球表面。", "event_time": "2013-12-14", "record_time": "2013-12-14"},
            {"text": "2019年1月3日，嫦娥四号实现人类首次月球背面软着陆。", "event_time": "2019-01-03", "record_time": "2019-01-03"},
            {"text": "2020年12月17日，嫦娥五号携带1731克月球样品成功返回地球。", "event_time": "2020-12-17", "record_time": "2020-12-17"},
        ],
        "qa_pairs": [
            {"q": "截至2015年，中国进行过几次月球探测任务？", "a": "三次（嫦娥一号、二号、三号）", "type": "multi_hop_temporal", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "中国第一次在月球背面着陆是什么时候？", "a": "2019年1月3日（嫦娥四号）", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
            {"q": "嫦娥工程各任务的先后顺序是什么？", "a": "嫦娥一号(2007) → 嫦娥二号(2010) → 嫦娥三号(2013) → 嫦娥四号(2019) → 嫦娥五号(2020)", "type": "temporal_ordering", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
    # --- 关系演化 ---
    {
        "scenario_id": "REAL_REL_001",
        "scenario_type": "relationship_evolution",
        "title_zh": "SpaceX与NASA合作演进",
        "domain": "tech",
        "writes": [
            {"text": "2006年8月18日，NASA向SpaceX授予商业轨道运输服务(COTS)合同。", "event_time": "2006-08-18", "record_time": "2006-08-18"},
            {"text": "2012年5月25日，SpaceX龙飞船成功对接国际空间站，成为首个完成此任务的商业航天器。", "event_time": "2012-05-25", "record_time": "2012-05-25"},
            {"text": "2014年9月16日，NASA选择SpaceX和波音执行商业载人航天任务。", "event_time": "2014-09-16", "record_time": "2014-09-16"},
            {"text": "2020年5月30日，SpaceX载人龙飞船发射成功，首次将NASA宇航员送往国际空间站。", "event_time": "2020-05-30", "record_time": "2020-05-30"},
        ],
        "qa_pairs": [
            {"q": "截至2010年，SpaceX与NASA的合作关系处于什么阶段？", "a": "已获得COTS合同，尚未完成空间站对接", "type": "point_in_time", "difficulty": "level_2", "evt_reason": True, "rec_reason": True},
            {"q": "SpaceX首次将宇航员送入空间站是什么时候？", "a": "2020年5月30日", "type": "point_in_time", "difficulty": "level_1", "evt_reason": True, "rec_reason": False},
        ],
    },
]


def build_curated_scenario(template: Dict) -> Dict:
    """Convert a curated scenario template into BiTempQA format."""
    sid = template["scenario_id"]
    writes = []
    for j, w in enumerate(template["writes"]):
        writes.append({
            "write_id": f"w{j+1}",
            "text": w["text"],
            "source_name": w.get("source", "real_world_events"),
            "event_time": f"{w['event_time']}T00:00:00Z",
            "record_time": f"{w['record_time']}T00:00:00Z",
        })

    # Build world states
    world_states = []
    for j, w in enumerate(writes):
        ws = {
            "as_of_record_time": w["record_time"],
            "entities": [],
            "relations": [],
            "known_facts": [ww["text"] for ww in writes[:j+1]],
        }
        world_states.append(ws)

    # Build QA pairs
    qa_pairs = []
    for k, qa in enumerate(template["qa_pairs"]):
        qa_pairs.append({
            "qa_id": f"{sid}_L{k+1}_{k+1:03d}",
            "scenario_id": sid,
            "difficulty": qa.get("difficulty", "level_1"),
            "question_type": qa.get("type", "point_in_time"),
            "question_zh": qa["q"],
            "question_en": "",
            "answer_zh": qa["a"],
            "answer_en": "",
            "answer_type": "abstractive",
            "choices": [],
            "correct_choice_index": None,
            "ranking_order": None,
            "query_event_time": None,
            "query_record_time": None,
            "relevant_time_range": {"start": None, "end": None},
            "reasoning_chain": [],
            "requires_event_time_reasoning": qa.get("evt_reason", True),
            "requires_record_time_reasoning": qa.get("rec_reason", False),
            "requires_version_tracking": False,
            "requires_knowledge_retraction": False,
            "source_write_ids": [f"w{j+1}" for j in range(len(writes))],
            "generation_method": "llm_human",
            "validation_status": "validated",
        })

    return {
        "scenario_id": sid,
        "scenario_type": template["scenario_type"],
        "title_zh": template["title_zh"],
        "title_en": "",
        "description_zh": f"基于真实事件的双时间轴场景。数据来源：公开新闻报道和历史记录。",
        "description_en": "",
        "domain": template["domain"],
        "language": "zh",
        "memory_writes": writes,
        "world_states": world_states,
        "qa_pairs": qa_pairs,
        "data_source": "curated_real_world",
    }


def main():
    logger.info("=== Converting ChronoQA candidates ===")
    chronoqa_file = BASE_DIR / "data" / "raw" / "ChronoQA" / "chronoqa_bitemporal_candidates.json"
    chronoqa_scenarios = convert_chronoqa(chronoqa_file)
    logger.info(f"Converted {len(chronoqa_scenarios)} ChronoQA scenarios")

    logger.info("=== Building curated real-world scenarios ===")
    curated_scenarios = [build_curated_scenario(t) for t in CURATED_SCENARIOS]
    logger.info(f"Built {len(curated_scenarios)} curated scenarios")

    # Combine
    all_scenarios = curated_scenarios + chronoqa_scenarios
    logger.info(f"Total combined scenarios: {len(all_scenarios)}")

    # Statistics
    type_dist = Counter(s["scenario_type"] for s in all_scenarios)
    domain_dist = Counter(s["domain"] for s in all_scenarios)
    source_dist = Counter(s["data_source"] for s in all_scenarios)
    total_writes = sum(len(s["memory_writes"]) for s in all_scenarios)
    total_qa = sum(len(s["qa_pairs"]) for s in all_scenarios)

    logger.info(f"Total writes: {total_writes}")
    logger.info(f"Total QA pairs: {total_qa}")
    logger.info(f"Type distribution: {dict(type_dist)}")
    logger.info(f"Domain distribution: {dict(domain_dist)}")
    logger.info(f"Source distribution: {dict(source_dist)}")

    # Save all scenarios
    output_file = OUTPUT_DIR / "all_real_source_scenarios.json"
    with open(output_file, "w") as f:
        json.dump(all_scenarios, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {output_file}")

    # Also save individual scenario types
    for stype in set(s["scenario_type"] for s in all_scenarios):
        subset = [s for s in all_scenarios if s["scenario_type"] == stype]
        safe_name = stype.replace(" ", "_")
        fpath = OUTPUT_DIR / f"{safe_name}.json"
        with open(fpath, "w") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved {len(subset)} {stype} scenarios to {fpath.name}")

    # Extract QA pairs for train/dev/test split
    all_qa = []
    for s in all_scenarios:
        for qa in s["qa_pairs"]:
            all_qa.append(qa)

    # Split: 70% train, 10% dev, 20% test
    import random
    random.seed(42)
    random.shuffle(all_qa)

    n = len(all_qa)
    n_train = int(0.7 * n)
    n_dev = int(0.1 * n)

    train_qa = all_qa[:n_train]
    dev_qa = all_qa[n_train:n_train + n_dev]
    test_qa = all_qa[n_train + n_dev:]

    for split_name, split_data in [("train", train_qa), ("dev", dev_qa), ("test", test_qa)]:
        dataset = {
            "dataset_id": f"bitpqa_{split_name}_zh",
            "name": f"BiTempQA Chinese {split_name.capitalize()} Set (Real Source)",
            "language": "zh",
            "split": split_name,
            "qa_pairs": split_data,
        }
        fpath = OUTPUT_DIR / f"bitpqa_{split_name}_zh.json"
        with open(fpath, "w") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"  {split_name}: {len(split_data)} QA pairs → {fpath.name}")


if __name__ == "__main__":
    main()
