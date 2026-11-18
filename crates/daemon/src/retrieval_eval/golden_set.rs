// SPDX-License-Identifier: Apache-2.0
//! The golden set: the seeded `CORPUS` and the `QUESTIONS` scored against it.
//! Question names and corpus labels are stable ids that reports quote: add
//! new ones, never rename.

use super::{Category, Fact, Knob, Question, DAY_MS};

const FACT_DEFAULTS: Fact = Fact {
    kind: "fact",
    label: "",
    content: "",
    scope: None,
    confidence: None,
    valid_from_offset_ms: None,
    edge_target: None,
    competes_with: None,
};

/// Fictional domain: Kestrel Robotics, a drone company, and its flagship
/// Falconer drone. Consistent across every entry so a reviewer can follow the
/// story; nothing here refers to a real person or company. Grouped under a
/// heading naming the category each block primarily serves; a handful of
/// facts (the `GraphExpansion` seeds) are also referenced by more than one
/// `Question`.
pub(super) const CORPUS: &[Fact] = &[
    // Lexical: exact/near-term-matching facts with no other knob involved.
    Fact {
        kind: "fact",
        label: "Kestrel HQ",
        content: "Kestrel Robotics headquarters is located in Boulder, Colorado.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Kestrel Founding",
        content: "Kestrel Robotics was founded in 2019 by Priya Anand and Tomas Reyes.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Titanium Frame Decision",
        content: "The engineering team chose a titanium frame for the Falconer drone to cut \
                   weight by 20 percent.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Battery",
        content: "The Falconer drone uses a 6400 mAh lithium battery pack.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Falconer Demo Day",
        content: "On demo day the Falconer drone completed a 45 minute autonomous flight \
                   without a single course correction.",
        ..FACT_DEFAULTS
    },
    // VectorSemantic: real-embedder tier only (WU-4); this WU just authors
    // the fact and its paraphrase-target `Question.query_text`.
    Fact {
        kind: "fact",
        label: "Solar Charging Pilot",
        content: "Kestrel Robotics is piloting solar panel charging docks for its warehouse \
                   drones.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Remote Work Policy",
        content: "Kestrel Robotics allows engineers to work from home three days a week.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Drone Swarm Research",
        content: "Kestrel Robotics researchers are studying how small drone swarms coordinate \
                   obstacle avoidance.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Customer Onboarding Redesign",
        content: "The customer success team redesigned onboarding to cut new customer setup \
                   time in half.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Noise Reduction Rotor",
        content: "The newest Falconer rotor blades reduce audible flight noise by 12 decibels.",
        ..FACT_DEFAULTS
    },
    // GraphExpansion: each seed is matched lexically by its Question; each
    // linked fact carries edge_target naming its seed, so the harness can
    // Graph::link them and prove expansion, not just lexical match, found it.
    Fact {
        kind: "decision",
        label: "Falconer Motor Recall",
        content: "Kestrel Robotics issued a recall for the Falconer's left motor after bearing \
                   wear was reported.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Motor Vendor Notice",
        content: "The motor vendor notified Kestrel Robotics about a batch of bearings that \
                   wear faster than spec.",
        edge_target: Some("Falconer Motor Recall"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Weatherproof Casing",
        content: "The engineering team weatherproofed the Falconer casing to withstand heavy \
                   rain.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Field Test Storm Report",
        content: "A field test flew the Falconer through a thunderstorm and the weatherproof \
                   casing kept all electronics dry.",
        edge_target: Some("Weatherproof Casing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Logistics Partnership Signed",
        content: "Kestrel Robotics signed a last mile delivery partnership with SwiftHaul.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "SwiftHaul Pilot Results",
        content: "The SwiftHaul delivery pilot completed 500 drone deliveries with a 98 \
                   percent on time rate.",
        edge_target: Some("Logistics Partnership Signed"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Software Update Rollback",
        content: "Kestrel Robotics rolled back the v3.2 flight control firmware update after \
                   reports of mid flight instability.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Firmware Bug Report",
        content: "A pilot reported the Falconer drifting sideways during hover after \
                   installing firmware v3.2.",
        edge_target: Some("Software Update Rollback"),
        ..FACT_DEFAULTS
    },
    // KindFilter: each pair shares the same topic terms and differs only in
    // kind, so a wrong kind filter would let the distractor rank instead.
    Fact {
        kind: "decision",
        label: "Warehouse Automation Decision",
        content: "Kestrel Robotics decided to move forward with warehouse automation using \
                   robotic picking arms.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Warehouse Automation Kickoff",
        content: "The warehouse automation project kicked off with a training session for \
                   warehouse staff.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Data Retention Policy",
        content: "Kestrel Robotics retains flight log data for two years before deletion.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Data Retention Incident",
        content: "An audit found flight log data retention exceeded two years before the \
                   policy was enforced.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Supplier Fact",
        content: "Kestrel Robotics sources lithium battery cells from a supplier based in \
                   Nevada.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Battery Supplier Switch Decision",
        content: "Kestrel Robotics decided to switch lithium battery cell suppliers to one \
                   based in Nevada.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Certification Flight Episode",
        content: "The certification flight for the Falconer drone passed all FAA test points \
                   on the first attempt.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Certification Flight Scheduling",
        content: "Kestrel Robotics scheduled the certification flight for the Falconer drone \
                   after final firmware review.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Insurance Provider Decision",
        content: "Kestrel Robotics chose a new insurance provider for its commercial drone \
                   fleet.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Insurance Provider Coverage",
        content: "The new insurance provider covers up to two million dollars per drone \
                   incident for the commercial fleet.",
        ..FACT_DEFAULTS
    },
    // ScopeFilter: same pattern, scope is the only thing that differs
    // between each pair.
    Fact {
        kind: "fact",
        label: "Engineering Sprint Cadence",
        content: "The engineering team runs two week sprint cycles for the Falconer firmware \
                   roadmap.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Marketing Sprint Cadence",
        content: "The marketing team also runs two week sprint cycles for campaign planning.",
        scope: Some("marketing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Legal Compliance Review",
        content: "Legal completed a compliance review of the Falconer's export control \
                   classification.",
        scope: Some("legal"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operations Compliance Review",
        content: "Operations completed a separate compliance review of warehouse safety \
                   procedures.",
        scope: Some("operations"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Finance Budget Approval",
        content: "Finance approved the annual budget for the Falconer production line.",
        scope: Some("finance"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Engineering Budget Request",
        content: "Engineering submitted a budget request for the Falconer production line \
                   expansion.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Operations Fleet Maintenance",
        content: "Operations scheduled fleet wide maintenance for all active Falconer drones.",
        scope: Some("operations"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Engineering Maintenance Design",
        content: "Engineering redesigned the maintenance access panel on the Falconer drone.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Marketing Launch Campaign",
        content: "Marketing launched a campaign announcing the Falconer drone to retail \
                   customers.",
        scope: Some("marketing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Legal Launch Review",
        content: "Legal reviewed the marketing materials before the Falconer drone launch \
                   campaign.",
        scope: Some("legal"),
        ..FACT_DEFAULTS
    },
    // Decay: each pair states the same fact, with the newer half restating
    // it plus one clause, so decay (not a lexical difference) is what should
    // separate them.
    Fact {
        kind: "fact",
        label: "Camera Sensor Spec (2024)",
        content: "The Falconer drone camera sensor captures 12 megapixel stills at 30 frames \
                   per second.",
        valid_from_offset_ms: Some(-500 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Camera Sensor Spec (2026)",
        content: "The Falconer drone camera sensor captures 12 megapixel stills at 30 frames \
                   per second, now with improved low light performance.",
        valid_from_offset_ms: Some(-2 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Range Spec (Early)",
        content: "The Falconer drone has a maximum flight range of 8 kilometers on a single \
                   charge.",
        valid_from_offset_ms: Some(-400 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Range Spec (Current)",
        content: "The Falconer drone has a maximum flight range of 8 kilometers on a single \
                   charge, extended by the new battery pack.",
        valid_from_offset_ms: Some(-3 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Support Hours (Old)",
        content: "Kestrel Robotics customer support operates from 9 AM to 5 PM Mountain Time.",
        valid_from_offset_ms: Some(-300 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Support Hours (New)",
        content: "Kestrel Robotics customer support operates from 9 AM to 5 PM Mountain Time, \
                   now with weekend coverage added.",
        valid_from_offset_ms: Some(-5 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Pricing Tier (Old)",
        content: "The Falconer drone starter package is priced at four thousand dollars.",
        valid_from_offset_ms: Some(-200 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Pricing Tier (New)",
        content: "The Falconer drone starter package is priced at four thousand dollars, \
                   unchanged from last year.",
        valid_from_offset_ms: Some(-DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Team Size (Old)",
        content: "The Kestrel Robotics engineering team has thirty employees.",
        valid_from_offset_ms: Some(-600 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Team Size (New)",
        content: "The Kestrel Robotics engineering team has thirty employees, following a \
                   recent hiring freeze.",
        valid_from_offset_ms: Some(-4 * DAY_MS),
        ..FACT_DEFAULTS
    },
    // AsOf: each pair is an original fact later superseded by a changed one;
    // the Question's Knob::AsOf offset sits between the two, so only the
    // original should be visible at that instant.
    Fact {
        kind: "fact",
        label: "Product Name (Original)",
        content: "The Falconer drone was originally named the SkyHawk before its public \
                   release.",
        valid_from_offset_ms: Some(-500 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Product Name (Renamed)",
        content: "The drone formerly called SkyHawk was renamed Falconer ahead of its public \
                   release.",
        valid_from_offset_ms: Some(-490 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Headquarters (Original)",
        content: "Kestrel Robotics' original headquarters was a shared workspace in Denver.",
        valid_from_offset_ms: Some(-800 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Headquarters (Current)",
        content: "Kestrel Robotics moved its headquarters from the shared Denver workspace to \
                   a dedicated site in Boulder.",
        valid_from_offset_ms: Some(-700 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "CEO (Original)",
        content: "Priya Anand served as Kestrel Robotics' first CEO.",
        valid_from_offset_ms: Some(-900 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "CEO (Current)",
        content: "Tomas Reyes became Kestrel Robotics' CEO after Priya Anand stepped down.",
        valid_from_offset_ms: Some(-100 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Product Line (Original)",
        content: "Kestrel Robotics originally built only agricultural survey drones.",
        valid_from_offset_ms: Some(-1000 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Product Line (Current)",
        content: "Kestrel Robotics expanded from agricultural survey drones into the Falconer \
                   commercial delivery line.",
        valid_from_offset_ms: Some(-300 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Motor Vendor (Original)",
        content: "The Falconer's motor was originally sourced from a supplier called Torque \
                   Dynamics.",
        valid_from_offset_ms: Some(-450 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Motor Vendor (Current)",
        content: "Kestrel Robotics switched the Falconer's motor supplier from Torque Dynamics \
                   to a new vendor after the bearing recall.",
        valid_from_offset_ms: Some(-50 * DAY_MS),
        ..FACT_DEFAULTS
    },
    // Confidence: each pair states the same fact from a verified and an
    // unverified source; competes_with on the higher-confidence half names
    // the one it must outrank.
    Fact {
        kind: "fact",
        label: "Falconer Top Speed (Verified)",
        content: "Verified wind tunnel testing measured the Falconer drone's top speed at 65 \
                   kilometers per hour.",
        confidence: Some(0.95),
        competes_with: Some("Falconer Top Speed (Field Report)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Falconer Top Speed (Field Report)",
        content: "An early field report estimated the Falconer drone's top speed at around 60 \
                   kilometers per hour.",
        confidence: Some(0.4),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Payload Capacity (Verified)",
        content: "Verified payload testing confirmed the Falconer drone carries up to 3 \
                   kilograms.",
        confidence: Some(0.9),
        competes_with: Some("Payload Capacity (Rumor)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Payload Capacity (Rumor)",
        content: "An unconfirmed rumor claimed the Falconer drone could carry up to 5 \
                   kilograms.",
        confidence: Some(0.3),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Life (Verified)",
        content: "Verified bench testing measured the Falconer drone's battery life at 40 \
                   minutes of flight.",
        confidence: Some(0.85),
        competes_with: Some("Battery Life (Marketing Estimate)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Life (Marketing Estimate)",
        content: "An early marketing estimate claimed the Falconer drone's battery life was 50 \
                   minutes of flight.",
        confidence: Some(0.5),
        ..FACT_DEFAULTS
    },
    // Expansion block: five more facts (or pairs) per category, appended after
    // the original corpus so existing labels never move. Every fact's content
    // is written to contain the answer to the question that names it, which is
    // what makes its relevance label correct by construction. Like the rest of
    // the corpus this is synthetic: LIAM is pre-release with no real usage or
    // query logs, so these questions stand in for the "real usage" set the
    // evaluation brief asks for, to be replaced once actual users exist.
    //
    // Lexical
    Fact {
        kind: "fact",
        label: "Delivery Network Size",
        content: "Kestrel Robotics operates 120 Falconer drones in its delivery network.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Harrier Survey Drone",
        content: "The Harrier survey drone maps crop fields using a multispectral camera.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Charging Dock Standard",
        content: "Kestrel Robotics adopted the Qi2 inductive standard for every charging dock.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Aerospace Expo Booth",
        content: "At the Aerospace Expo in Denver the company booth drew 3000 visitors in two \
                   days.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Operator Certificate Policy",
        content: "Every Falconer operator must hold a Part 107 remote pilot certificate.",
        ..FACT_DEFAULTS
    },
    // VectorSemantic: each fact is the sole real semantic match for its
    // paraphrase and shares almost no vocabulary with it.
    Fact {
        kind: "fact",
        label: "Night Flight Waiver",
        content: "Kestrel Robotics received regulatory approval to operate its drones after \
                   sunset.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Employee Equity Plan",
        content: "Kestrel Robotics grants equity options to every full time employee after one \
                   year.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Cybersecurity Audit",
        content: "An external firm tested the drone control software for hacking \
                   vulnerabilities and found none critical.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Recycled Packaging",
        content: "Shipping boxes for Falconer spare parts are now made entirely from recycled \
                   cardboard.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Wildlife Disturbance Study",
        content: "A study found the drone flight paths rarely disturbed nesting birds near the \
                   Boulder test field.",
        ..FACT_DEFAULTS
    },
    // GraphExpansion: five more seed and linked pairs, same shape as above.
    Fact {
        kind: "decision",
        label: "Antenna Upgrade Decision",
        content: "Engineers replaced the telemetry antenna with a longer range model.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Ridge Signal Trial",
        content: "A ridge top trial confirmed the signal stayed solid at twelve kilometers.",
        edge_target: Some("Antenna Upgrade Decision"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Gimbal Redesign Decision",
        content: "The team redesigned the camera gimbal to stop vibration blur in footage.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Crisp Inspection Images",
        content: "Customers reported crisp inspection images after the stabilizer update \
                   shipped.",
        edge_target: Some("Gimbal Redesign Decision"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Training Academy Launch",
        content: "Kestrel Robotics opened a pilot training academy in Austin.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Academy First Cohort",
        content: "Twenty four students graduated the inaugural class with perfect safety scores.",
        edge_target: Some("Training Academy Launch"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Cloud Telemetry Migration",
        content: "Flight telemetry storage moved from on premise servers to a managed cloud \
                   database.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Hosting Cost Savings",
        content: "Hosting expenses dropped by 35 percent within one quarter.",
        edge_target: Some("Cloud Telemetry Migration"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Propeller Order Pause",
        content: "Kestrel Robotics paused orders from its propeller manufacturer over missed \
                   delivery dates.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Backup Blade Vendor",
        content: "A second manufacturer in Ohio agreed to supply carbon fiber blades within \
                   six weeks.",
        edge_target: Some("Propeller Order Pause"),
        ..FACT_DEFAULTS
    },
    // KindFilter: five more pairs sharing topic terms and differing in kind.
    Fact {
        kind: "decision",
        label: "Hangar Lease Decision",
        content: "Kestrel Robotics decided to lease a second hangar near the Boulder airfield.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Hangar Lease Walkthrough",
        content: "Leadership toured the second hangar near the Boulder airfield last week.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Battery Disposal Policy",
        content: "Retired batteries must be recycled through a certified hazardous waste handler.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Battery Disposal Incident",
        content: "A retired battery sent to a standard waste handler overheated in transit.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Hospital Escalation Episode",
        content: "A hospital customer escalated a delayed delivery complaint to the executive \
                   team.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Hospital Escalation Decision",
        content: "The executive team decided to assign a dedicated manager to hospital customer \
                   delivery complaints.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Gust Tolerance Fact",
        content: "The Falconer safely operates in sustained gusts up to forty kilometers per \
                   hour.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "policy",
        label: "Gust Grounding Policy",
        content: "Operators must ground the Falconer when sustained gusts exceed thirty \
                   kilometers per hour.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "decision",
        label: "Open Source Firmware Decision",
        content: "Kestrel Robotics decided to publish the ground station firmware as open \
                   source.",
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "episode",
        label: "Open Source Firmware Announcement",
        content: "The ground station firmware repository went public at a community meetup, \
                   open source at last.",
        ..FACT_DEFAULTS
    },
    // ScopeFilter: five more pairs differing only in scope.
    Fact {
        kind: "fact",
        label: "Support Ticket Backlog",
        content: "Support cleared the ticket backlog after adding two weekend shifts.",
        scope: Some("support"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Sales Ticket Backlog",
        content: "Sales reviewed the ticket backlog impact on renewal conversations.",
        scope: Some("sales"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "HR Hiring Plan",
        content: "HR opened twelve new roles for the firmware team this year.",
        scope: Some("hr"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Engineering Hiring Needs",
        content: "Engineering listed the firmware skills missing from the current team roles.",
        scope: Some("engineering"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Security Access Audit",
        content: "Security audited badge access to the drone assembly hall.",
        scope: Some("security"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operations Badge Procedure",
        content: "Operations updated the badge access procedure for assembly hall visitors.",
        scope: Some("operations"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Sales Pipeline Review",
        content: "Sales reviewed the enterprise pipeline for delivery drone contracts.",
        scope: Some("sales"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Marketing Case Studies",
        content: "Marketing drafted case studies from the enterprise delivery drone contracts.",
        scope: Some("marketing"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Finance Tax Credit",
        content: "Finance filed for a research tax credit covering drone development costs.",
        scope: Some("finance"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Legal Tax Credit Deadline",
        content: "Legal confirmed the research tax credit filing met the statutory deadline.",
        scope: Some("legal"),
        ..FACT_DEFAULTS
    },
    // Decay: five more pairs, the newer half restating the older plus a
    // clause.
    Fact {
        kind: "fact",
        label: "Warranty Term (Old)",
        content: "Falconer drones carry a one year warranty on all flight components.",
        valid_from_offset_ms: Some(-450 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Warranty Term (New)",
        content: "Falconer drones carry a one year warranty on all flight components, now \
                   extended to cover battery packs.",
        valid_from_offset_ms: Some(-4 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Charge Time (Old)",
        content: "A depleted Falconer battery reaches full charge in ninety minutes.",
        valid_from_offset_ms: Some(-380 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Charge Time (New)",
        content: "A depleted Falconer battery reaches full charge in ninety minutes, helped by \
                   the faster charging dock.",
        valid_from_offset_ms: Some(-3 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Depot Count (Old)",
        content: "Kestrel Robotics runs three regional depots for drone maintenance.",
        valid_from_offset_ms: Some(-350 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Depot Count (New)",
        content: "Kestrel Robotics runs three regional depots for drone maintenance, with \
                   another site under construction.",
        valid_from_offset_ms: Some(-2 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "App Rating (Old)",
        content: "The Kestrel mobile app holds a four point five star store rating.",
        valid_from_offset_ms: Some(-280 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "App Rating (New)",
        content: "The Kestrel mobile app holds a four point five star store rating, with over \
                   ten thousand reviews.",
        valid_from_offset_ms: Some(-4 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Lead Time (Old)",
        content: "New Falconer orders ship within six weeks of purchase.",
        valid_from_offset_ms: Some(-420 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Lead Time (New)",
        content: "New Falconer orders ship within six weeks of purchase, now tracked in the \
                   customer portal.",
        valid_from_offset_ms: Some(-DAY_MS),
        ..FACT_DEFAULTS
    },
    // AsOf: five more pairs, an original fact later superseded.
    Fact {
        kind: "fact",
        label: "Support Channel (Original)",
        content: "Customers originally reached support only by phone.",
        valid_from_offset_ms: Some(-600 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Support Channel (Current)",
        content: "Kestrel Robotics replaced phone only support with a chat portal.",
        valid_from_offset_ms: Some(-200 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Assembly Site (Original)",
        content: "The first Falconer assembly line ran from a leased warehouse in Longmont.",
        valid_from_offset_ms: Some(-700 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Assembly Site (Current)",
        content: "Assembly moved from the leased Longmont warehouse into the new Boulder \
                   factory.",
        valid_from_offset_ms: Some(-250 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Funding Round (Original)",
        content: "Kestrel Robotics closed a two million dollar seed round.",
        valid_from_offset_ms: Some(-650 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Funding Round (Current)",
        content: "Kestrel Robotics followed its seed round with a twenty million dollar \
                   Series A.",
        valid_from_offset_ms: Some(-300 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Flight Controller (Original)",
        content: "Early Falconer prototypes flew on an off the shelf open source autopilot \
                   board.",
        valid_from_offset_ms: Some(-550 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Flight Controller (Current)",
        content: "Kestrel Robotics replaced the open source autopilot board with an in house \
                   flight controller.",
        valid_from_offset_ms: Some(-150 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Delivery Radius (Original)",
        content: "Falconer deliveries were limited to a five kilometer radius around each \
                   depot.",
        valid_from_offset_ms: Some(-480 * DAY_MS),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Delivery Radius (Current)",
        content: "The delivery radius around each depot grew from five kilometers to eight.",
        valid_from_offset_ms: Some(-90 * DAY_MS),
        ..FACT_DEFAULTS
    },
    // Confidence: five more pairs; the verified half names the unverified
    // one it must outrank.
    Fact {
        kind: "fact",
        label: "Charge Cycles (Verified)",
        content: "Verified lab cycling showed the Falconer battery lasts 800 charge cycles.",
        confidence: Some(0.92),
        competes_with: Some("Charge Cycles (Forum Post)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Charge Cycles (Forum Post)",
        content: "A forum post claimed the Falconer battery lasts 1500 charge cycles.",
        confidence: Some(0.35),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operating Temperature (Verified)",
        content: "Chamber testing verified Falconer operation from minus ten to forty degrees \
                   Celsius.",
        confidence: Some(0.9),
        competes_with: Some("Operating Temperature (Blog)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Operating Temperature (Blog)",
        content: "A blog post said the Falconer works down to minus thirty degrees Celsius.",
        confidence: Some(0.3),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Open Orders (Verified)",
        content: "The finance ledger shows 340 open Falconer orders.",
        confidence: Some(0.88),
        competes_with: Some("Open Orders (Hallway Rumor)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Open Orders (Hallway Rumor)",
        content: "Hallway chatter suggested 900 open Falconer orders.",
        confidence: Some(0.25),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Crash Rate (Verified)",
        content: "Flight logs verified a crash rate of one per ten thousand flights.",
        confidence: Some(0.93),
        competes_with: Some("Crash Rate (Anecdote)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Crash Rate (Anecdote)",
        content: "An anecdote from one customer implied a crash rate of one per thousand \
                   flights.",
        confidence: Some(0.4),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Ingress Rating (Verified)",
        content: "The certification report verified an IP55 ingress protection rating.",
        confidence: Some(0.95),
        competes_with: Some("Ingress Rating (Sales Slide)"),
        ..FACT_DEFAULTS
    },
    Fact {
        kind: "fact",
        label: "Ingress Rating (Sales Slide)",
        content: "An old sales slide advertised an IP67 ingress protection rating.",
        confidence: Some(0.45),
        ..FACT_DEFAULTS
    },
];

pub(super) const QUESTIONS: &[Question] = &[
    // Lexical
    Question {
        name: "lexical_hq_location",
        category: Category::Lexical,
        query_text: "Where is Kestrel Robotics headquarters located?",
        knob: Knob::None,
        expected_relevant: &["Kestrel HQ"],
        answer: "Boulder, Colorado",
    },
    Question {
        name: "lexical_founders",
        category: Category::Lexical,
        query_text: "Who founded Kestrel Robotics?",
        knob: Knob::None,
        expected_relevant: &["Kestrel Founding"],
        answer: "Priya Anand and Tomas Reyes",
    },
    Question {
        name: "lexical_titanium_frame",
        category: Category::Lexical,
        query_text: "What frame material did the engineering team choose for the Falconer \
                      drone?",
        knob: Knob::None,
        expected_relevant: &["Titanium Frame Decision"],
        answer: "titanium frame",
    },
    Question {
        name: "lexical_falconer_battery",
        category: Category::Lexical,
        query_text: "What battery does the Falconer drone use?",
        knob: Knob::None,
        expected_relevant: &["Falconer Battery"],
        answer: "6400 mAh",
    },
    Question {
        name: "lexical_demo_day_flight",
        category: Category::Lexical,
        query_text: "How long did the Falconer drone fly autonomously on demo day?",
        knob: Knob::None,
        expected_relevant: &["Falconer Demo Day"],
        answer: "45 minute",
    },
    // VectorSemantic (real-embedder tier only, WU-4): each query_text is a
    // paraphrase with weak lexical overlap against its expected fact.
    Question {
        name: "vector_solar_charging",
        category: Category::VectorSemantic,
        query_text: "Is the company experimenting with any renewable energy for recharging \
                      its fleet?",
        knob: Knob::None,
        expected_relevant: &["Solar Charging Pilot"],
        answer: "solar panel charging",
    },
    Question {
        name: "vector_remote_work",
        category: Category::VectorSemantic,
        query_text: "Can staff do their jobs from a home office part of the week?",
        knob: Knob::None,
        expected_relevant: &["Remote Work Policy"],
        answer: "three days a week",
    },
    Question {
        name: "vector_drone_swarm",
        category: Category::VectorSemantic,
        query_text: "What is being studied about groups of small flying robots avoiding \
                      obstacles together?",
        knob: Knob::None,
        expected_relevant: &["Drone Swarm Research"],
        answer: "obstacle avoidance",
    },
    Question {
        name: "vector_onboarding_speed",
        category: Category::VectorSemantic,
        query_text: "Did the company make it quicker for new clients to get started?",
        knob: Knob::None,
        expected_relevant: &["Customer Onboarding Redesign"],
        answer: "cut new customer setup time in half",
    },
    Question {
        name: "vector_quieter_rotors",
        category: Category::VectorSemantic,
        query_text: "Are the blades on the newest aircraft quieter than before?",
        knob: Knob::None,
        expected_relevant: &["Noise Reduction Rotor"],
        answer: "12 decibels",
    },
    // GraphExpansion: query_text surfaces the seed via lexical match;
    // expected_relevant names both the seed and its linked fact.
    Question {
        name: "graph_motor_recall_reason",
        category: Category::GraphExpansion,
        // Deliberately avoids "Kestrel Robotics": that phrase appears in
        // most of CORPUS (company-history/org facts), and at HARNESS_K's
        // widened k, fts5_query's stopword-OR would then lexically match
        // the entire corpus for this query, leaving nothing for
        // GraphExpansion to discover that wasn't already found lexically
        // (see the strengthened per-question assertion below, which
        // caught exactly this before this comment was added).
        query_text: "Why was the Falconer's motor recalled?",
        knob: Knob::None,
        expected_relevant: &["Falconer Motor Recall", "Motor Vendor Notice"],
        answer: "bearings that wear faster than spec",
    },
    Question {
        name: "graph_weatherproof_casing",
        category: Category::GraphExpansion,
        query_text: "How did the engineering team weatherproof the Falconer casing?",
        knob: Knob::None,
        expected_relevant: &["Weatherproof Casing", "Field Test Storm Report"],
        answer: "kept all electronics dry",
    },
    Question {
        name: "graph_logistics_partnership",
        category: Category::GraphExpansion,
        query_text: "What logistics partnership did Kestrel Robotics sign?",
        knob: Knob::None,
        expected_relevant: &["Logistics Partnership Signed", "SwiftHaul Pilot Results"],
        answer: "98 percent on time",
    },
    Question {
        name: "graph_firmware_rollback",
        category: Category::GraphExpansion,
        query_text: "Why was the v3.2 firmware update rolled back?",
        knob: Knob::None,
        expected_relevant: &["Software Update Rollback", "Firmware Bug Report"],
        answer: "drifting sideways",
    },
    Question {
        name: "graph_motor_bearing_defect",
        category: Category::GraphExpansion,
        query_text: "What bearing defect caused the Falconer motor recall?",
        knob: Knob::None,
        expected_relevant: &["Falconer Motor Recall", "Motor Vendor Notice"],
        answer: "wear faster than spec",
    },
    // KindFilter
    Question {
        name: "kind_warehouse_decision",
        category: Category::KindFilter,
        query_text: "warehouse automation",
        knob: Knob::Kind("decision"),
        expected_relevant: &["Warehouse Automation Decision"],
        answer: "robotic picking arms",
    },
    Question {
        name: "kind_data_retention_policy",
        category: Category::KindFilter,
        query_text: "flight log data retention",
        knob: Knob::Kind("policy"),
        expected_relevant: &["Data Retention Policy"],
        answer: "two years",
    },
    Question {
        name: "kind_battery_supplier_fact",
        category: Category::KindFilter,
        query_text: "lithium battery cell supplier Nevada",
        knob: Knob::Kind("fact"),
        expected_relevant: &["Battery Supplier Fact"],
        answer: "Nevada",
    },
    Question {
        name: "kind_certification_flight_episode",
        category: Category::KindFilter,
        query_text: "certification flight Falconer drone",
        knob: Knob::Kind("episode"),
        expected_relevant: &["Certification Flight Episode"],
        answer: "passed all FAA test points",
    },
    Question {
        name: "kind_insurance_provider_decision",
        category: Category::KindFilter,
        query_text: "insurance provider commercial drone fleet",
        knob: Knob::Kind("decision"),
        expected_relevant: &["Insurance Provider Decision"],
        answer: "new insurance provider",
    },
    // ScopeFilter
    Question {
        name: "scope_engineering_sprint",
        category: Category::ScopeFilter,
        query_text: "two week sprint cycles",
        knob: Knob::Scope("engineering"),
        expected_relevant: &["Engineering Sprint Cadence"],
        answer: "two week sprint cycles",
    },
    Question {
        name: "scope_legal_compliance",
        category: Category::ScopeFilter,
        query_text: "compliance review",
        knob: Knob::Scope("legal"),
        expected_relevant: &["Legal Compliance Review"],
        answer: "export control classification",
    },
    Question {
        name: "scope_finance_budget",
        category: Category::ScopeFilter,
        query_text: "Falconer production line budget",
        knob: Knob::Scope("finance"),
        expected_relevant: &["Finance Budget Approval"],
        answer: "approved the annual budget",
    },
    Question {
        name: "scope_operations_maintenance",
        category: Category::ScopeFilter,
        query_text: "Falconer maintenance",
        knob: Knob::Scope("operations"),
        expected_relevant: &["Operations Fleet Maintenance"],
        answer: "fleet wide maintenance",
    },
    Question {
        name: "scope_marketing_launch",
        category: Category::ScopeFilter,
        query_text: "Falconer drone launch campaign",
        knob: Knob::Scope("marketing"),
        expected_relevant: &["Marketing Launch Campaign"],
        answer: "launched a campaign",
    },
    // Decay
    Question {
        name: "decay_camera_sensor",
        category: Category::Decay,
        query_text: "Falconer drone camera sensor megapixel stills frames per second",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Camera Sensor Spec (2026)"],
        answer: "12 megapixel",
    },
    Question {
        name: "decay_flight_range",
        category: Category::Decay,
        query_text: "Falconer drone maximum flight range single charge",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Range Spec (Current)"],
        answer: "8 kilometers",
    },
    Question {
        name: "decay_support_hours",
        category: Category::Decay,
        query_text: "Kestrel Robotics customer support operating hours Mountain Time",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Support Hours (New)"],
        answer: "9 AM to 5 PM",
    },
    Question {
        name: "decay_pricing_tier",
        category: Category::Decay,
        query_text: "Falconer drone starter package price",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Pricing Tier (New)"],
        answer: "four thousand dollars",
    },
    Question {
        name: "decay_team_size",
        category: Category::Decay,
        query_text: "Kestrel Robotics engineering team employees",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Team Size (New)"],
        answer: "thirty employees",
    },
    // AsOf
    Question {
        name: "asof_product_name",
        category: Category::AsOf,
        query_text: "What was the Falconer drone originally called before its public \
                      release?",
        knob: Knob::AsOf(-495 * DAY_MS),
        expected_relevant: &["Product Name (Original)"],
        answer: "SkyHawk",
    },
    Question {
        name: "asof_headquarters",
        category: Category::AsOf,
        query_text: "Where was Kestrel Robotics' original headquarters before the move to \
                      Boulder?",
        knob: Knob::AsOf(-750 * DAY_MS),
        expected_relevant: &["Headquarters (Original)"],
        answer: "Denver",
    },
    Question {
        name: "asof_ceo",
        category: Category::AsOf,
        query_text: "Who was Kestrel Robotics' first CEO?",
        knob: Knob::AsOf(-500 * DAY_MS),
        expected_relevant: &["CEO (Original)"],
        answer: "Priya Anand",
    },
    Question {
        name: "asof_product_line",
        category: Category::AsOf,
        query_text: "What kind of drones did Kestrel Robotics originally build?",
        knob: Knob::AsOf(-600 * DAY_MS),
        expected_relevant: &["Product Line (Original)"],
        answer: "agricultural survey drones",
    },
    Question {
        name: "asof_motor_vendor",
        category: Category::AsOf,
        query_text: "Which vendor originally supplied the Falconer's motor?",
        knob: Knob::AsOf(-250 * DAY_MS),
        expected_relevant: &["Falconer Motor Vendor (Original)"],
        answer: "Torque Dynamics",
    },
    // Confidence: query_text matches both halves of a pair; expected_relevant
    // names the higher-confidence half, whose competes_with names the loser
    // (checked below).
    Question {
        name: "confidence_top_speed_terms",
        category: Category::Confidence,
        query_text: "Falconer drone top speed kilometers per hour",
        knob: Knob::None,
        expected_relevant: &["Falconer Top Speed (Verified)"],
        answer: "65 kilometers per hour",
    },
    Question {
        name: "confidence_top_speed_phrase",
        category: Category::Confidence,
        query_text: "How fast can the Falconer drone fly?",
        knob: Knob::None,
        expected_relevant: &["Falconer Top Speed (Verified)"],
        answer: "65 kilometers per hour",
    },
    Question {
        name: "confidence_payload_terms",
        category: Category::Confidence,
        query_text: "Falconer drone payload capacity kilograms",
        knob: Knob::None,
        expected_relevant: &["Payload Capacity (Verified)"],
        answer: "3 kilograms",
    },
    Question {
        name: "confidence_payload_phrase",
        category: Category::Confidence,
        query_text: "How much weight can the Falconer drone carry?",
        knob: Knob::None,
        expected_relevant: &["Payload Capacity (Verified)"],
        answer: "3 kilograms",
    },
    Question {
        name: "confidence_battery_life",
        category: Category::Confidence,
        query_text: "Falconer drone battery life minutes of flight",
        knob: Knob::None,
        expected_relevant: &["Battery Life (Verified)"],
        answer: "40 minutes",
    },
    // Expansion block: five more questions per category, each naming the
    // facts the matching expansion block in CORPUS wrote to answer it.
    //
    // Lexical
    Question {
        name: "lexical_delivery_network_size",
        category: Category::Lexical,
        query_text: "How many drones are in the delivery network?",
        knob: Knob::None,
        expected_relevant: &["Delivery Network Size"],
        answer: "120 Falconer",
    },
    Question {
        name: "lexical_harrier_survey",
        category: Category::Lexical,
        query_text: "Which drone maps crop fields with a multispectral camera?",
        knob: Knob::None,
        expected_relevant: &["Harrier Survey Drone"],
        answer: "multispectral camera",
    },
    Question {
        name: "lexical_charging_dock_standard",
        category: Category::Lexical,
        query_text: "Which inductive standard do the charging docks use?",
        knob: Knob::None,
        expected_relevant: &["Charging Dock Standard"],
        answer: "Qi2",
    },
    Question {
        name: "lexical_expo_visitors",
        category: Category::Lexical,
        query_text: "How many visitors came to the company booth at the Aerospace Expo?",
        knob: Knob::None,
        expected_relevant: &["Aerospace Expo Booth"],
        answer: "3000 visitors",
    },
    Question {
        name: "lexical_operator_certificate",
        category: Category::Lexical,
        query_text: "What certificate must a Falconer operator hold?",
        knob: Knob::None,
        expected_relevant: &["Operator Certificate Policy"],
        answer: "Part 107",
    },
    // VectorSemantic
    Question {
        name: "vector_night_flight",
        category: Category::VectorSemantic,
        query_text: "Is the firm allowed to fly its aircraft in the dark?",
        knob: Knob::None,
        expected_relevant: &["Night Flight Waiver"],
        answer: "after sunset",
    },
    Question {
        name: "vector_equity_plan",
        category: Category::VectorSemantic,
        query_text: "Do workers get a share of ownership in the business after a while?",
        knob: Knob::None,
        expected_relevant: &["Employee Equity Plan"],
        answer: "after one year",
    },
    Question {
        name: "vector_security_audit",
        category: Category::VectorSemantic,
        query_text: "Did outsiders probe the flight software for ways attackers could break in?",
        knob: Knob::None,
        expected_relevant: &["Cybersecurity Audit"],
        answer: "none critical",
    },
    Question {
        name: "vector_recycled_packaging",
        category: Category::VectorSemantic,
        query_text: "Has the packaging for replacement components become more eco friendly?",
        knob: Knob::None,
        expected_relevant: &["Recycled Packaging"],
        answer: "recycled cardboard",
    },
    Question {
        name: "vector_wildlife_impact",
        category: Category::VectorSemantic,
        query_text: "Does flying the machines upset animals that live near the testing site?",
        knob: Knob::None,
        expected_relevant: &["Wildlife Disturbance Study"],
        answer: "rarely disturbed nesting birds",
    },
    // GraphExpansion
    Question {
        name: "graph_antenna_upgrade",
        category: Category::GraphExpansion,
        query_text: "telemetry antenna replaced",
        knob: Knob::None,
        expected_relevant: &["Antenna Upgrade Decision", "Ridge Signal Trial"],
        answer: "twelve kilometers",
    },
    Question {
        name: "graph_gimbal_redesign",
        category: Category::GraphExpansion,
        query_text: "camera gimbal redesigned",
        knob: Knob::None,
        expected_relevant: &["Gimbal Redesign Decision", "Crisp Inspection Images"],
        answer: "crisp inspection images",
    },
    Question {
        name: "graph_training_academy",
        category: Category::GraphExpansion,
        query_text: "Austin pilot training",
        knob: Knob::None,
        expected_relevant: &["Training Academy Launch", "Academy First Cohort"],
        answer: "Twenty four students",
    },
    Question {
        name: "graph_cloud_migration",
        category: Category::GraphExpansion,
        query_text: "flight telemetry storage managed cloud database",
        knob: Knob::None,
        expected_relevant: &["Cloud Telemetry Migration", "Hosting Cost Savings"],
        answer: "35 percent",
    },
    Question {
        name: "graph_propeller_pause",
        category: Category::GraphExpansion,
        query_text: "Why were propeller orders paused?",
        knob: Knob::None,
        expected_relevant: &["Propeller Order Pause", "Backup Blade Vendor"],
        answer: "carbon fiber blades",
    },
    // KindFilter
    Question {
        name: "kind_hangar_decision",
        category: Category::KindFilter,
        query_text: "second hangar Boulder airfield",
        knob: Knob::Kind("decision"),
        expected_relevant: &["Hangar Lease Decision"],
        answer: "second hangar",
    },
    Question {
        name: "kind_battery_disposal_policy",
        category: Category::KindFilter,
        query_text: "retired batteries hazardous waste handler",
        knob: Knob::Kind("policy"),
        expected_relevant: &["Battery Disposal Policy"],
        answer: "certified hazardous waste handler",
    },
    Question {
        name: "kind_hospital_escalation_episode",
        category: Category::KindFilter,
        query_text: "hospital customer delivery complaint",
        knob: Knob::Kind("episode"),
        expected_relevant: &["Hospital Escalation Episode"],
        answer: "delayed delivery complaint",
    },
    Question {
        name: "kind_gust_tolerance_fact",
        category: Category::KindFilter,
        query_text: "sustained gusts Falconer",
        knob: Knob::Kind("fact"),
        expected_relevant: &["Gust Tolerance Fact"],
        answer: "forty kilometers per hour",
    },
    Question {
        name: "kind_open_source_episode",
        category: Category::KindFilter,
        query_text: "ground station firmware open source",
        knob: Knob::Kind("episode"),
        expected_relevant: &["Open Source Firmware Announcement"],
        answer: "went public",
    },
    // ScopeFilter
    Question {
        name: "scope_support_backlog",
        category: Category::ScopeFilter,
        query_text: "ticket backlog",
        knob: Knob::Scope("support"),
        expected_relevant: &["Support Ticket Backlog"],
        answer: "two weekend shifts",
    },
    Question {
        name: "scope_hr_hiring",
        category: Category::ScopeFilter,
        query_text: "firmware team roles",
        knob: Knob::Scope("hr"),
        expected_relevant: &["HR Hiring Plan"],
        answer: "twelve new roles",
    },
    Question {
        name: "scope_security_badge",
        category: Category::ScopeFilter,
        query_text: "badge access assembly hall",
        knob: Knob::Scope("security"),
        expected_relevant: &["Security Access Audit"],
        answer: "badge access",
    },
    Question {
        name: "scope_sales_contracts",
        category: Category::ScopeFilter,
        query_text: "enterprise delivery drone contracts",
        knob: Knob::Scope("sales"),
        expected_relevant: &["Sales Pipeline Review"],
        answer: "enterprise pipeline",
    },
    Question {
        name: "scope_legal_tax_credit",
        category: Category::ScopeFilter,
        query_text: "research tax credit",
        knob: Knob::Scope("legal"),
        expected_relevant: &["Legal Tax Credit Deadline"],
        answer: "statutory deadline",
    },
    // Decay
    Question {
        name: "decay_warranty_term",
        category: Category::Decay,
        query_text: "Falconer warranty flight components",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Warranty Term (New)"],
        answer: "one year warranty",
    },
    Question {
        name: "decay_charge_time",
        category: Category::Decay,
        query_text: "Falconer battery full charge ninety minutes",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Charge Time (New)"],
        answer: "ninety minutes",
    },
    Question {
        name: "decay_depot_count",
        category: Category::Decay,
        query_text: "Kestrel Robotics three regional depots for drone maintenance",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Depot Count (New)"],
        answer: "three regional depots",
    },
    Question {
        name: "decay_app_rating",
        category: Category::Decay,
        query_text: "Kestrel mobile app store rating",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["App Rating (New)"],
        answer: "four point five star",
    },
    Question {
        name: "decay_lead_time",
        category: Category::Decay,
        query_text: "New Falconer orders ship within six weeks of purchase",
        knob: Knob::HalfLife(30 * DAY_MS),
        expected_relevant: &["Lead Time (New)"],
        answer: "six weeks",
    },
    // AsOf
    Question {
        name: "asof_support_channel",
        category: Category::AsOf,
        query_text: "How did customers originally reach support?",
        knob: Knob::AsOf(-400 * DAY_MS),
        expected_relevant: &["Support Channel (Original)"],
        answer: "only by phone",
    },
    Question {
        name: "asof_assembly_site",
        category: Category::AsOf,
        query_text: "Where did the first Falconer assembly line run?",
        knob: Knob::AsOf(-500 * DAY_MS),
        expected_relevant: &["Assembly Site (Original)"],
        answer: "Longmont",
    },
    Question {
        name: "asof_funding_round",
        category: Category::AsOf,
        query_text: "How large was the seed round Kestrel Robotics closed?",
        knob: Knob::AsOf(-500 * DAY_MS),
        expected_relevant: &["Funding Round (Original)"],
        answer: "two million dollar",
    },
    Question {
        name: "asof_flight_controller",
        category: Category::AsOf,
        query_text: "Which autopilot board did early Falconer prototypes use?",
        knob: Knob::AsOf(-350 * DAY_MS),
        expected_relevant: &["Flight Controller (Original)"],
        answer: "open source autopilot",
    },
    Question {
        name: "asof_delivery_radius",
        category: Category::AsOf,
        query_text: "What was the delivery radius around each depot originally?",
        knob: Knob::AsOf(-300 * DAY_MS),
        expected_relevant: &["Delivery Radius (Original)"],
        answer: "five kilometer",
    },
    // Confidence
    Question {
        name: "confidence_charge_cycles",
        category: Category::Confidence,
        query_text: "Falconer battery charge cycles",
        knob: Knob::None,
        expected_relevant: &["Charge Cycles (Verified)"],
        answer: "800 charge cycles",
    },
    Question {
        name: "confidence_operating_temperature",
        category: Category::Confidence,
        query_text: "Falconer operating temperature degrees Celsius",
        knob: Knob::None,
        expected_relevant: &["Operating Temperature (Verified)"],
        answer: "minus ten to forty degrees",
    },
    Question {
        name: "confidence_open_orders",
        category: Category::Confidence,
        query_text: "How many open Falconer orders are there?",
        knob: Knob::None,
        expected_relevant: &["Open Orders (Verified)"],
        answer: "340",
    },
    Question {
        name: "confidence_crash_rate",
        category: Category::Confidence,
        query_text: "What is the Falconer crash rate per flights?",
        knob: Knob::None,
        expected_relevant: &["Crash Rate (Verified)"],
        answer: "one per ten thousand",
    },
    Question {
        name: "confidence_ingress_rating",
        category: Category::Confidence,
        query_text: "Falconer ingress protection rating",
        knob: Knob::None,
        expected_relevant: &["Ingress Rating (Verified)"],
        answer: "IP55",
    },
];
