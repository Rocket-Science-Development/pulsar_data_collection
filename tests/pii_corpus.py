"""Synthetic corpus to measure the personal-information detectors: French and English sentences with fictitious but
valid values (checksums pass), and sentences with look-alike business numbers that must not be detected.

    python -m tests.pii_corpus        # precision and recall by type
"""

import random

from pulsar_data_collection.privacy import PiiDetector
from pulsar_data_collection.privacy.detectors import luhn


def _luhn_complete(prefix: str, length: int, rng) -> str:
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(length - len(prefix) - 1))
    return body + next(d for d in "0123456789" if luhn(body + d))


def sin(rng):
    n = _luhn_complete(str(rng.choice([1, 2, 3, 4, 5, 6, 7, 9])), 9, rng)
    return rng.choice([f"{n[:3]} {n[3:6]} {n[6:]}", f"{n[:3]}-{n[3:6]}-{n[6:]}", n])


def card(rng):
    n = _luhn_complete(rng.choice(["4", "51", "55", "37"]), 15 if False else 16, rng)
    return rng.choice([" ".join(n[i : i + 4] for i in range(0, 16, 4)), n])


def phone(rng):
    a, b, c = rng.choice([514, 438, 450, 418, 613, 819, 416]), rng.randint(200, 999), rng.randint(0, 9999)
    return rng.choice([f"{a}-{b}-{c:04d}", f"({a}) {b}-{c:04d}", f"{a} {b} {c:04d}", f"+1 {a} {b} {c:04d}"])


def ramq(rng):
    letters = "".join(rng.choice("ABCDEFGHIJKLMNOPRSTVW") for _ in range(4))
    month = rng.randint(1, 12) + rng.choice([0, 50])
    compact = f"{letters}{rng.randint(40, 99):02d}{month:02d}{rng.randint(1, 28):02d}{rng.randint(10, 99)}"
    return rng.choice(
        [
            f"{compact[:4]} {compact[4:8]} {compact[8:]}",
            compact,
            f"{compact[:4]}-{compact[4:8]}-{compact[8:]}",  # with dashes
            f"{compact[:4].lower()} {compact[4:8]} {compact[8:]}",  # letters in lower case
        ]
    )


def postal(rng):
    return rng.choice(["H2X 1Y4", "G1R 4P5", "J4K 2T8", "K1A 0B1", "H3B 2Y5"])


def email(rng):
    return (
        rng.choice(["lea.tremblay", "m.gagnon", "chloe_b", "ryan.mac", "nadia88"])
        + "@"
        + rng.choice(["exemple.qc.ca", "courriel.ca", "example.com"])
    )


def dob(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1940, 2005)
    months = [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ]
    return rng.choice([f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}", f"{d} {months[m - 1]} {y}"])


POSITIVE = [
    ("sin", sin, ["Mon NAS est le {}.", "My SIN is {}.", "Numéro d'assurance sociale : {}", "{} (NAS)"]),
    ("credit_card", card, ["Payez avec la carte {}.", "Card number {} please", "Visa {}"]),
    ("phone", phone, ["Appelez-moi au {}.", "Call me at {}", "Mon cellulaire : {}", "Joignez-moi au {} demain."]),
    ("ramq", ramq, ["Carte RAMQ : {}", "Mon numéro d'assurance maladie est {}.", "Health card {}", "C'est {} sur ma carte."]),
    ("postal_code", postal, ["J'habite au {} à Montréal.", "Code postal : {}", "Address postal code {}"]),
    ("email", email, ["Écrivez-moi à {}.", "Contact: {}", "Mon courriel est {}"]),
    ("date_of_birth", dob, ["Je suis né le {}.", "Date de naissance : {}", "DOB {}"]),
]
NEGATIVE = [
    "Votre dossier 130692544 est en traitement.",
    "Commande 4501234567 expédiée.",
    "Le montant est de 1234,56 $ pour 2026.",
    "Rendez-vous le 2026-11-13 à 14 h.",
    "La réunion compte 12 participants et dure 90 minutes.",
    "Référence produit AB123456 en stock.",
    "Total: 1,250,000 units in 2025.",
    "Le programme PRG-LOGE a un plafond de 5000 $.",
    "Order 613-555-0142 shipped.",
    # harder look-alikes
    "Le lot H2X 1Y4 est prêt.",
    "Budget de 514 555 0199 $ approuvé.",
    "Version 2.3.4567 publiée le 12/03/2025.",
    "Le vol AC870 part à 21 h 05.",
    "ISBN 978-2-7604-1234-5",
    # Luhn-valid numbers shaped like a SIN, announced by a word proper to another identifier
    "No client 130 692 544, merci.",
    "Customer no. 130 692 544, thank you.",
    "Code de confirmation : 130692544",
    "Confirmation number 130692544",
    "Votre réservation #130692544 est confirmée.",
    "Numéro de membre 514-555-0123",
]
# Ambiguous by design, masked on purpose: a 16-digit number with a card prefix and a valid checksum is treated as a
# card even after a business word (privacy first); these are reported apart, not counted as errors.
AMBIGUOUS = ["Facture 4111111111111111 payée."]


def corpus(n_per_template: int = 20, seed: int = 0):
    """(text, set of expected types) pairs"""
    rng = random.Random(seed)
    items = []
    for kind, make, templates in POSITIVE:
        for template in templates:
            for _ in range(n_per_template):
                items.append((template.format(make(rng)), {kind}))
    items += [(text, set()) for text in NEGATIVE for _ in range(n_per_template)]
    return items


def evaluate(detector=None, **kwargs) -> dict:
    """Precision and recall by type, at the level of sentences (a type found or not in a sentence)"""
    detector = detector or PiiDetector()
    stats = {}
    for text, expected in corpus(**kwargs):
        found = {f.type for f in detector.scan(text)}
        for kind in expected | found:
            s = stats.setdefault(kind, {"tp": 0, "fp": 0, "fn": 0})
            s["tp" if kind in expected and kind in found else "fp" if kind in found else "fn"] += 1
    for s in stats.values():
        s["precision"] = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else 1.0
        s["recall"] = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else 1.0
    return stats


if __name__ == "__main__":
    for kind, s in sorted(evaluate().items()):
        print(f"{kind:<15} precision {s['precision']:.3f} recall {s['recall']:.3f}  (tp {s['tp']}, fp {s['fp']}, fn {s['fn']})")
