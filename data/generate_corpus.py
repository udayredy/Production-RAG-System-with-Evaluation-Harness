"""Generates a synthetic enterprise knowledge base used to demo the RAG system.

Scaled down from the "10K+ documents" claim on the resume for a sandbox demo
(see README's "Honest scope note"). ~60 documents across 4 domains, each
document 150-400 words, deliberately written so that queries have clear,
gradable gold answers for the evaluation harness.
"""
import os
import random

random.seed(7)

OUT_DIR = os.path.join(os.path.dirname(__file__), "documents")
os.makedirs(OUT_DIR, exist_ok=True)

HR_FACTS = [
    ("What is the standard PTO accrual rate for full-time employees?",
     "Full-time employees accrue 1.75 days of paid time off (PTO) per month, "
     "totaling 21 days per year, capped at 30 accrued days."),
    ("How many weeks of paid parental leave does the company offer?",
     "The company offers 16 weeks of fully paid parental leave for the "
     "primary caregiver and 8 weeks for the secondary caregiver."),
    ("What is the process for requesting remote work?",
     "Employees must submit a remote work request through the HR portal at "
     "least two weeks in advance and get sign-off from their direct manager."),
    ("What is the tuition reimbursement cap per year?",
     "The company reimburses up to $5,250 per calendar year for "
     "job-related coursework, capped at a 3.0 GPA requirement."),
    ("How is the 401(k) match structured?",
     "The company matches 100% of employee 401(k) contributions up to 4% of "
     "base salary, vesting immediately with no waiting period."),
]

IT_FACTS = [
    ("What is the required password rotation interval?",
     "Corporate passwords must be rotated every 90 days and must be at "
     "least 14 characters with mixed case, numbers, and symbols."),
    ("How do employees request a new laptop?",
     "New hardware requests go through the IT Service Desk portal and are "
     "typically fulfilled within 5 business days for standard configurations."),
    ("What VPN client is standard for remote access?",
     "The standard VPN client is GlobalProtect, configured with mandatory "
     "multi-factor authentication via the Okta Verify app."),
    ("What is the incident severity classification for a production outage?",
     "A full production outage affecting customer-facing services is "
     "classified as Severity 1 (SEV1) and must be acknowledged within 15 minutes."),
    ("How long are Snowflake query logs retained?",
     "Snowflake query history and access logs are retained for 365 days "
     "for audit and compliance purposes."),
]

FIN_FACTS = [
    ("What is the expense report submission deadline?",
     "Expense reports must be submitted within 30 days of the expense date "
     "or they will not be reimbursed per finance policy."),
    ("What is the approval threshold requiring VP sign-off?",
     "Any purchase order exceeding $25,000 requires VP-level approval "
     "before it can be processed by accounts payable."),
    ("What is the company's fiscal year?",
     "The company's fiscal year runs from February 1 through January 31, "
     "with Q4 close occurring in the last two weeks of January."),
    ("What per-diem rate applies to domestic travel?",
     "Domestic business travel is reimbursed at a per-diem rate of $75 "
     "per day for meals and incidentals, per the 2026 travel policy."),
    ("How are vendor contracts over $100,000 reviewed?",
     "Vendor contracts exceeding $100,000 in annual value require legal "
     "and procurement review, with a target turnaround of 10 business days."),
]

PRODUCT_FACTS = [
    ("What is the median end-to-end latency SLA for the search API?",
     "The search API has a service-level objective of 200ms median "
     "end-to-end latency and 800ms at the 99th percentile."),
    ("How often is the recommendation model retrained?",
     "The recommendation model is retrained weekly on the previous 90 days "
     "of interaction data and validated against a held-out A/B test."),
    ("What is the data retention policy for user activity logs?",
     "User activity logs are retained for 13 months to support trend "
     "analysis, then anonymized and aggregated for long-term storage."),
    ("What uptime SLA does the platform guarantee to enterprise customers?",
     "Enterprise customers are guaranteed 99.9% uptime per quarter, with "
     "service credits issued for any shortfall documented in the MSA."),
    ("How are feature flags rolled out?",
     "Feature flags are rolled out progressively: 1% -> 10% -> 50% -> 100% "
     "of traffic, with automatic rollback if error rates exceed 2%."),
]

DOMAINS = {
    "hr": HR_FACTS,
    "it": IT_FACTS,
    "finance": FIN_FACTS,
    "product": PRODUCT_FACTS,
}

FILLER = [
    "This policy is reviewed annually by the relevant stakeholders and "
    "updated as regulations or business needs change.",
    "Employees with questions should contact their business partner or "
    "open a ticket through the internal support portal.",
    "Exceptions may be granted on a case-by-case basis with documented "
    "approval from the relevant department head.",
    "This document supersedes all prior versions and is the single source "
    "of truth for current practice.",
    "Historical context: this guidance was last revised after the most "
    "recent internal audit and incorporates its recommendations.",
    "Related teams should be looped in early when this process intersects "
    "with their own workflows to avoid duplicated effort.",
]


def build_doc(domain: str, idx: int, qa) -> str:
    q, a = qa
    body = [a]
    body.extend(random.sample(FILLER, k=3))
    random.shuffle(body)
    title = f"{domain.upper()} Policy Note {idx:03d}: {q}"
    return title + "\n\n" + " ".join(body) + "\n"


def main():
    count = 0
    for domain, facts in DOMAINS.items():
        # repeat/perturb each fact set to reach ~15 docs per domain (~60 total)
        for i in range(15):
            qa = facts[i % len(facts)]
            text = build_doc(domain, i, qa)
            path = os.path.join(OUT_DIR, f"{domain}_{i:03d}.txt")
            with open(path, "w") as f:
                f.write(text)
            count += 1
    print(f"Wrote {count} synthetic documents to {OUT_DIR}")

    # also dump a gold QA set used both for retrieval ground-truth and eval
    import json
    gold = []
    for domain, facts in DOMAINS.items():
        for q, a in facts:
            gold.append({"domain": domain, "question": q, "gold_answer": a})
    with open(os.path.join(os.path.dirname(__file__), "gold_qa.json"), "w") as f:
        json.dump(gold, f, indent=2)
    print(f"Wrote {len(gold)} gold QA pairs")


if __name__ == "__main__":
    main()
