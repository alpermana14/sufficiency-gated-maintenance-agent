"""Gate v2 - the labelled query set the retrieval-sufficiency gate is scored on.

Answers CAEE R1-8 and R2-18: the first submission calibrated and tested the gate on the
same nine queries. Here the set is larger, written against the store of build_store.py
before any distance was computed, and split into a development half used to choose the
distance floor and a test half used only to report.

LABEL (author decision, 16 Sep 2026; see build_store.py):
  sufficient = True   a record in the store materially answers the query;
  sufficient = False  the store only touches the topic without answering it
                      (partially relevant), or does not cover it at all.
  Contradictory records count as sufficient: the evidence exists, and resolving the
  conflict is a separate mechanism.

CATEGORIES
  in_coverage         a record answers the question directly
  paraphrase          the same request in other words, informally or with a typo
  partially_relevant  the store names the topic but not the detail asked for
  near_irrelevant     a neighbouring subsystem or symptom the store does not cover
  out_of_domain       not about this conveyor at all
  contradictory       two records disagree on the fix; the evidence still exists
  terminology         technician wording and local short forms for covered cases
"""
import json
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries.json")
# Author verification of a 40-item sample on 17 Sep 2026 agreed with 37 labels and corrected
# three. Each correction applies to the case, so it covers the canonical query and its
# paraphrase: pair 24 (the records recommend inspecting the belt edge but record no action) and
# pair 40 (the record says the take-up travel was to be verified, not what was found).
PAIRS_INSUFFICIENT = {24, 40}
DEV_FRACTION = 0.40
SPLIT_SEED = 20260916

# (canonical query, paraphrase, gold record ids)
PAIRS = [
    ("Have we had belt mistracking on this conveyor before?",
     "belt keeps wandering to one side, has that happened before?", ["WO-03", "WO-14"]),
    ("What was done the last time the belt slipped under load?",
     "belt slipped when it was loaded, what did we do last time?", ["WO-06"]),
    ("Is there a past record of jerky running at start-up?",
     "conveyor judders on start up, any history?", ["WO-04"]),
    ("Have we replaced an emergency stop button before?",
     "have we ever swapped out an e-stop button?", ["WO-08"]),
    ("What caused the frequency inverter to trip on overcurrent?",
     "why did the inverter keep tripping on overcurrent?", ["WO-09"]),
    ("Has the accelerometer cable ever caused a missing vibration reading?",
     "vibration reading went flat once, was that the sensor cable?", ["WO-10"]),
    ("What did we do about coupling misalignment at the drive end?",
     "drive end coupling was out of line, what did we do?", ["WO-11"]),
    ("Have we topped up the gearbox oil before?",
     "have we ever had to put oil in the gearbox?", ["WO-12"]),
    ("Was an idler roller ever replaced because it seized?",
     "did we ever change an idler that had seized up?", ["WO-13"]),
    ("Which grease is used on the roller bearings?",
     "what grease goes on the roller bearings?", ["WO-15"]),
    ("Has product carry-back on the return side been dealt with?",
     "stuff sticking on the return belt, did we sort that out?", ["WO-16"]),
    ("Were loose motor mount bolts ever found on this machine?",
     "any history of loose motor mounting bolts?", ["WO-17"]),
    ("Has the guard interlock switch given a false open signal before?",
     "guard switch said open when it wasnt, seen that before?", ["WO-18"]),
    ("What explained the rise in acoustic noise without a vibration change?",
     "got noisier but vibration stayed flat, what was that?", ["WO-19"]),
    ("Is there a record about a slow start after a weekend shutdown?",
     "slow to start after the weekend, is that written up anywhere?", ["WO-20"]),
    ("Have we had a distribution-shift episode caused by extra load?",
     "ever had an alarm that turned out to be just extra load?", ["WO-07"]),
    ("What was recommended when the vibration reached ISO zone C?",
     "when we hit zone C what did they tell us to do?", ["WO-07"]),
    ("Has debris under the return rollers caused a noise complaint?",
     "rubbish under the return rollers making noise, any record?", ["WO-02"]),
    ("Was the drive chain ever re-tensioned?",
     "have we ever tightened the drive chain?", ["WO-04"]),
    ("Have the idler rollers been lubricated because they ran dry?",
     "idlers went dry once and got greased, right?", ["WO-05"]),
    ("What preventive maintenance has been recorded for this conveyor?",
     "whats the routine maintenance we have on record?", ["WO-01", "WO-15", "WO-20"]),
    ("Which past work orders mention the tail pulley?",
     "which jobs mention the tail pulley?", ["WO-03", "WO-14"]),
    ("Has a scraper blade been replaced on this machine?",
     "did we ever put a new scraper blade on?", ["WO-16"]),
    ("What did we do when the belt edge risked fraying?",
     "belt edge was about to fray, what did we do?", ["WO-02", "WO-03"]),
    ("Have we ever extended the inverter acceleration ramp?",
     "did we ever stretch the accel ramp on the drive?", ["WO-09"]),
    ("Was any work order raised for a safety device on this conveyor?",
     "any job raised on the safety gear?", ["WO-08", "WO-18"]),
    ("Which record describes a seal that was losing oil?",
     "which job was the one with the leaking seal?", ["WO-12"]),
    ("Have we seen a rise in z_rms that was not caused by load?",
     "z_rms crept up with no load change, happened before?", ["WO-17"]),
    ("What action was taken for a squealing idler set?",
     "idlers were squealing, what got done?", ["WO-13"]),
    ("Is there a past case where no mechanical intervention was needed?",
     "ever had a case where nothing needed fixing?", ["WO-07", "WO-20"]),
    ("Has the belt been re-tensioned after another repair?",
     "did we retension the belt after some other repair?", ["WO-06", "WO-14"]),
    ("Which work orders were given High priority?",
     "which jobs were marked high priority?", ["WO-08", "WO-18"]),
    ("Have we recorded a fault found during a weekly function test?",
     "did the weekly function test ever turn up a fault?", ["WO-08"]),
    ("What was done about dust getting into the rollers?",
     "dust got into the rollers, what did we do?", ["WO-13"]),
    ("Has a loose cover ever been the cause of a noise complaint?",
     "was a loose cover ever the reason for noise?", ["WO-19"]),
    ("Which record explains a belt drift that came back after repair?",
     "belt drifted again after the fix, which job was that?", ["WO-14"]),
    ("Have we adjusted tracking bolts on this conveyor?",
     "have we touched the tracking bolts before?", ["WO-03"]),
    ("Was the motor current ever monitored after a repair?",
     "did we watch the motor current after a repair?", ["WO-06"]),
    ("Has a sensor connector been secured with a cable tie?",
     "did someone cable tie the sensor connector?", ["WO-10"]),
    ("What was recorded about the take-up travel of the belt?",
     "is there anything written about the take-up travel?", ["WO-06"]),
]

# The store names the topic but not the detail asked for -> the gate must report a gap,
# because letting these through is exactly how invented detail enters an answer.
PARTIALLY_RELEVANT = [
    "What torque value was used on the motor mount bolts?",
    "Which grease was applied during the January lubrication round?",
    "How many hours of downtime did the seized idler roller cause?",
    "Who carried out the belt re-tensioning in April?",
    "What belt tension value was set in April, in newtons?",
    "What was the vibration reading when the drive chain was re-tensioned?",
    "How much had the drive chain stretched in March, in per cent?",
    "Which brand of scraper blade was fitted in September?",
    "What is the rated load capacity of this conveyor?",
    "How many decibels was the noise after the inspection cover was refitted?",
    "What part number was used for the replacement emergency stop button?",
    "How long did the inverter ramp change take to implement?",
    "What was the ambient temperature when the gearbox oil was topped up?",
    "How many litres of gearbox oil were added?",
    "What was the belt speed during the load-change event?",
    "Which shift supervisor approved the March work order?",
    "What was the x_rms value during the belt drift in March?",
    "How many idler rollers are installed on this conveyor?",
    "What is the service interval for the roller lubrication?",
    "Which tool was used to realign the coupling?",
    "What dial gauge reading was recorded after the coupling realignment?",
    "How many grease nipples are there on the drive end?",
    "What was the motor current before the belt was re-tensioned?",
    "Which supplier provided the replacement idler roller?",
    "What was the cost of the crowned tail pulley replacement?",
    "How long did the conveyor stop for the scraper blade change?",
    "What is the wear limit of the scraper blade, in millimetres?",
    "What was the guard interlock switch model?",
    "How many times had the inverter tripped before the ramp was changed?",
    "What was the exact time the accelerometer cable was reseated?",
    "Which cleaning method was used for the frame debris?",
    "What is the acceptable z_rms baseline for this conveyor, in mm/s?",
    "How often is the dust shield inspected?",
    "What was the oil level reading before the gearbox top-up?",
    "Which fastener size is used for the motor mount bolts?",
]

# A neighbouring subsystem or symptom the store does not cover at all.
NEAR_IRRELEVANT = [
    "What was done the last time a bearing outer race failed?",
    "Have we had bearing spalling on the drive shaft?",
    "Is there a record of a motor winding insulation failure?",
    "What did the gearbox oil analysis show for contamination?",
    "Have we repaired a belt splice failure before?",
    "Has the drive pulley lagging ever been replaced due to wear?",
    "When was the temperature sensor last calibrated?",
    "Has the gearbox itself ever been replaced?",
    "Is there a record of a crack in the conveyor frame?",
    "Have we had a bearing temperature alarm on this machine?",
    "What was done when the motor overheated?",
    "Have we replaced the drive motor?",
    "Is there a past record of a shaft fracture?",
    "What did we do about rotor imbalance on the motor?",
    "Have we had a gear tooth failure in the gearbox?",
    "Is there a record of a belt burn-through?",
    "What was done about corrosion on the frame?",
    "Have we had a lubrication pump failure?",
    "Is there a record of a control cabinet fan failure?",
    "Have we replaced the drive belt of the motor fan?",
    "What was done when the encoder failed?",
    "Is there a record of a contactor welding shut?",
    "Have we had water ingress into the motor terminal box?",
    "What did we do about a cracked idler shaft?",
    "Is there a record of a pulley bearing seizure?",
    "Have we had a brake failure on this conveyor?",
    "What was done about a loose foundation bolt of the frame?",
    "Is there a record of belt elongation beyond the take-up range?",
    "Have we had a failure of the safety relay?",
    "What did we do when the chain sprocket teeth wore out?",
]

# Not about this conveyor at all.
OUT_OF_DOMAIN = [
    "What is the maintenance history of the packaging machine?",
    "Show me the work orders for the forklift.",
    "What did we do to the air compressor last month?",
    "Is there a record of a pump seal replacement in the utility room?",
    "What maintenance was done on conveyor line 2?",
    "Have we serviced the overhead crane this year?",
    "What is the history of the injection moulding machine?",
    "Show me past work orders for the cooling tower.",
    "What did we do about the boiler feedwater pump?",
    "Is there a record of chiller maintenance?",
    "What faults were logged on the palletiser?",
    "Have we repaired the shrink wrapper before?",
    "What is the service history of the plant generator?",
    "Show me the records for the warehouse racking inspection.",
    "What was done to the workshop lathe?",
    "Is there a work order for the office air conditioning?",
    "What maintenance is due on the fire pump?",
    "Have we had a fault on the weighbridge?",
    "What did we do about the dust collector fan?",
    "Show me the calibration records of the laboratory balance.",
    "What is the repair history of the delivery van?",
    "Have we serviced the hydraulic press?",
    "What faults were recorded on the labelling machine?",
    "Is there a record of the sprinkler system test?",
    "What was done to the plant water treatment unit?",
]

# Two records disagree on the fix for the same symptom. Evidence exists, so the gate
# must let these through; resolving the conflict is a separate mechanism.
CONTRADICTORY = [
    ("What is the correct fix when the belt drifts to the drive side?", ["WO-03", "WO-14"]),
    ("Should I adjust the tracking bolts if the belt drifts again?", ["WO-03", "WO-14"]),
    ("Do our records agree on how to correct belt mistracking?", ["WO-03", "WO-14"]),
    ("Has the advice on belt drift changed over time?", ["WO-03", "WO-14"]),
    ("What did we try first for belt mistracking, and did it hold?", ["WO-03", "WO-14"]),
    ("Which repair finally stopped the belt from drifting?", ["WO-03", "WO-14"]),
    ("Is replacing the tail pulley recommended for belt drift?", ["WO-14"]),
    ("Is re-tensioning enough when the belt drifts to one side?", ["WO-03", "WO-06", "WO-14"]),
    ("How many times has belt drift been reported on this conveyor?", ["WO-03", "WO-14"]),
    ("What is the most recent recommendation for belt mistracking?", ["WO-14"]),
    ("Did the March repair for belt drift work?", ["WO-03", "WO-14"]),
    ("Were the tracking bolts adjusted more than once?", ["WO-03", "WO-14"]),
    ("Which record should I follow for a drifting belt, the older or the newer one?",
     ["WO-03", "WO-14"]),
    ("Has a worn crowned pulley been diagnosed on this conveyor?", ["WO-14"]),
    ("Do we have conflicting maintenance advice for any symptom?", ["WO-03", "WO-14"]),
]

# Technician wording and local short forms for cases the store does cover.
TERMINOLOGY = [
    ("belt wandering off to one side, any history?", ["WO-03", "WO-14"], True),
    ("chains gone slack, happened before?", ["WO-04"], True),
    ("e-stop button ever packed up?", ["WO-08"], True),
    ("VFD tripped, any record of that?", ["WO-09"], True),
    ("what grease for the roller nipples?", ["WO-15"], True),
    ("belt slipping, what fixed it last time?", ["WO-06"], True),
    ("idler squealing, ever dealt with?", ["WO-13"], True),
    ("vibe sensor lead ever come loose?", ["WO-10"], True),
    ("coupling knocking, anything on record?", ["WO-11"], True),
    ("gearbox low on oil before?", [], False),  # author check: WO-12 is the incident itself, not a prior one
    ("motor feet bolts ever found loose?", ["WO-17"], True),
    ("any history of a spalled bearing here?", [], False),
    ("motor windings ever burnt out?", [], False),
    ("loose cover causing a racket, any record?", ["WO-19"], True),
    ("scraper worn out, ever changed?", ["WO-16"], True),
]


def build():
    items = []

    prefix = {"in_coverage": "IC", "paraphrase": "PP", "partially_relevant": "PR",
              "near_irrelevant": "NI", "out_of_domain": "OD", "contradictory": "CD",
              "terminology": "TM"}

    def add(cat, text, sufficient, gold, pair_id=None):
        items.append({"id": f"{prefix[cat]}-{sum(1 for i in items if i['category'] == cat) + 1:03d}",
                      "category": cat, "query": text, "sufficient": sufficient,
                      "gold_records": gold, "pair_id": pair_id})

    for k, (canon, para, gold) in enumerate(PAIRS, start=1):
        suff = k not in PAIRS_INSUFFICIENT
        g = gold if suff else []
        add("in_coverage", canon, suff, g, pair_id=f"P{k:02d}")
        add("paraphrase", para, suff, g, pair_id=f"P{k:02d}")
    for q in PARTIALLY_RELEVANT:
        add("partially_relevant", q, False, [])
    for q in NEAR_IRRELEVANT:
        add("near_irrelevant", q, False, [])
    for q in OUT_OF_DOMAIN:
        add("out_of_domain", q, False, [])
    for q, gold in CONTRADICTORY:
        add("contradictory", q, True, gold)
    for q, gold, suff in TERMINOLOGY:
        add("terminology", q, suff, gold)

    # stratified split by category; paraphrase pairs stay on the same side
    rng = random.Random(SPLIT_SEED)
    by_cat = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    dev_pairs = set()
    pair_ids = sorted({it["pair_id"] for it in items if it["pair_id"]})
    rng.shuffle(pair_ids)
    dev_pairs.update(pair_ids[: round(len(pair_ids) * DEV_FRACTION)])
    for cat, group in by_cat.items():
        if group[0]["pair_id"]:
            for it in group:
                it["split"] = "dev" if it["pair_id"] in dev_pairs else "test"
        else:
            idx = list(range(len(group)))
            rng.shuffle(idx)
            dev = set(idx[: round(len(group) * DEV_FRACTION)])
            for j, it in enumerate(group):
                it["split"] = "dev" if j in dev else "test"
    return items


def main():
    items = build()
    assert len(items) == 200, len(items)
    assert len({i["id"] for i in items}) == len(items)
    payload = {"dev_fraction": DEV_FRACTION, "split_seed": SPLIT_SEED, "queries": items}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(items)} queries to {OUT}")
    for cat in ["in_coverage", "paraphrase", "partially_relevant", "near_irrelevant",
                "out_of_domain", "contradictory", "terminology"]:
        g = [i for i in items if i["category"] == cat]
        d = sum(1 for i in g if i["split"] == "dev")
        s = sum(1 for i in g if i["sufficient"])
        print(f"  {cat:20s} n={len(g):3d}  sufficient={s:3d}  dev={d:3d}  test={len(g) - d:3d}")
    print(f"  {'TOTAL':20s} n={len(items):3d}  sufficient={sum(1 for i in items if i['sufficient']):3d}"
          f"  dev={sum(1 for i in items if i['split'] == 'dev'):3d}"
          f"  test={sum(1 for i in items if i['split'] == 'test'):3d}")


if __name__ == "__main__":
    main()
