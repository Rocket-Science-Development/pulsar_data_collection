"""Limits found by the night lab (LP-01, LP-02) on the rule-based personal-information detectors, and their fixes"""

import random

import pandas as pd
import pytest

from pulsar_data_collection.privacy import PiiDetector, PiiPolicy, PiiSession
from pulsar_data_collection.privacy.detectors import luhn

SIN = "130 692 544"  # fictitious, valid checksum
FINANCE_TYPES = ("iban", "bic", "bank_account", "bank_routing", "card_security_code")


def types(detector, text):
    return [f.type for f in detector.scan(text)]


def _luhn_sin(rng) -> str:
    body = str(rng.choice("1234567")) + "".join(str(rng.randint(0, 9)) for _ in range(7))
    number = body + next(d for d in "0123456789" if luhn(body + d))
    return f"{number[:3]} {number[3:6]} {number[6:]}"


# ---- C4: business words hid leaks in the outputs of agents


def test_leak_after_an_earlier_mention_of_a_file_is_counted():
    with PiiSession() as session:
        session.scan("input", "Bonjour, où en est ma demande ?")
        leak = session.scan("output", f"Le dossier voisin indique aussi {SIN}.")
        assert [f.type for f in leak.leaks] == ["sin"]
        neighbour = session.scan("output", "Your file was reviewed; the neighbour can be reached at 418 555 0142.")
        assert [f.type for f in neighbour.leaks] == ["phone"]
    with PiiSession(output_business_scope="window") as session:  # the previous behaviour, on request
        assert session.scan("output", f"Le dossier voisin indique aussi {SIN}.").leaks == []
    with pytest.raises(ValueError):
        PiiSession(output_business_scope="sentence")
    with pytest.raises(ValueError):
        PiiDetector().scan("text", business_scope="sentence")


def test_business_numbers_in_outputs_are_not_leaks():
    """The narrow rule (business word right before the value) creates no false leak on Luhn-valid look-alikes"""
    rng = random.Random(7)
    outputs = []
    for _ in range(300):
        s9 = _luhn_sin(rng)
        compact = s9.replace(" ", "")
        outputs += [
            f"Votre dossier {s9} est en traitement.",
            f"Your case {s9} is being processed.",
            f"Dossier n° {s9} : complet.",
            f"Référence : {s9}",
            f"Code de confirmation : {compact}",
            f"Confirmation number {compact}",
            f"No client {s9}, merci.",
            f"Customer no. {s9}, thank you.",
            f"Commande {rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)} expédiée hier.",
            f"Facture TREM{rng.randint(10, 99)}0{rng.randint(1, 9)}{rng.randint(10, 28)}{rng.randint(10, 99)} payée.",
        ]
    with PiiSession() as session:
        session.scan("input", "Pouvez-vous vérifier mon dossier ?")
        leaks = [text for text in outputs if session.scan("output", text).leaks]
    assert leaks == []


def test_values_the_user_gave_near_a_business_word_are_not_leaks():
    """Inputs are scanned both ways, so that a value hidden by a business word in the input is still 'provided'"""
    with PiiSession() as session:
        given = session.scan("input", f"Concernant le dossier de ma mère, son numéro est {SIN}")
        assert given.findings == []  # inputs are counted as before
        echo = session.scan("output", f"Le numéro {SIN} est bien enregistré.")
        assert [f.type for f in echo.findings] == ["sin"] and echo.leaks == []


def test_session_with_a_detector_without_scopes():
    class Plain:
        def scan(self, text):
            return PiiDetector().scan(text)

    with PiiSession(Plain()) as session:
        assert session.scan("output", f"Votre dossier {SIN}").leaks == []
        assert [f.type for f in session.scan("output", f"NAS {SIN}").leaks] == ["sin"]


# ---- M17: RAMQ with dashes or in lower case


def test_ramq_with_dashes_or_in_lower_case():
    detector = PiiDetector()
    for value in ("TREM 8551 1512", "TREM85511512", "TREM-8551-1512", "trem 8551 1512", "trem-8551-1512", "Trem85511512"):
        assert types(detector, f"Carte soleil {value}") == ["ramq"], value
        assert types(detector, f"C'est {value} sur ma carte.") == ["ramq"], value
    assert types(detector, "Carte soleil trem-8513-1512") == []  # month 13 still rejected
    lower = detector.scan("C'est trem 8551 1512 sur ma carte.")[0].confidence
    assert lower < detector.scan("C'est TREM 8551 1512.")[0].confidence
    with PiiSession() as session:
        session.scan("input", "Ma carte soleil : TREM 8551 1512")
        assert session.scan("output", "J'ai noté trem-8551-1512.").leaks == []  # same value, other format


# ---- M18: Luhn-valid customer numbers and confirmation codes reported as SIN


def test_luhn_valid_customer_and_confirmation_numbers_are_not_sins():
    detector = PiiDetector()
    compact = SIN.replace(" ", "")
    for text in (
        f"No client {SIN}, merci.",
        f"Customer no. {SIN}, thank you.",
        f"Code de confirmation : {compact}",
        f"Confirmation number {compact}",
        f"Votre réservation #{compact} est confirmée.",
        f"Code #{compact}",
    ):
        assert types(detector, text) == [], text
    assert types(detector, f"Le NAS du client : {SIN}") == ["sin"]  # the SIN word wins
    assert types(detector, f"Je vous confirme le {SIN} pour la demande.") == ["sin"]  # "confirme" is not "confirmation"
    assert types(detector, "Numéro de membre 514-555-0123") == []  # a generic "numéro" loses to "membre" right before
    assert types(detector, "Mon numéro : 514-555-0123") == ["phone"]
    assert types(detector, "Membre depuis 2019, appelez-moi au 514-555-0123") == ["phone"]


# ---- M15: formats of other countries, behind `locales`


def test_other_countries_are_opt_in():
    texts = {
        "ssn_us": "SSN 123-45-6789",
        "phone": "Call me at +44 20 7946 0958",
        "nir_fr": "Numéro de sécurité sociale : 1 85 05 78 006 084 91",
        "nino_uk": "National Insurance AB 12 34 56 C",
        "passport": "Passport number 123456789",
        "drivers_licence": "Driver's license D1234567",
    }
    default = PiiDetector()
    every = PiiDetector(locales=("ca", "us", "fr", "uk", "intl"))
    for kind, text in texts.items():
        assert types(default, text) == [], text
        assert types(every, text) == [kind], text
    fr = PiiDetector(locales=("ca", "fr"))
    assert types(fr, "Appelez-moi au 06 12 34 56 78") == ["phone"]
    assert types(fr, "Numéro de sécurité sociale : 1 85 05 78 006 084 92") == []  # wrong key
    us = PiiDetector(locales=("us",))
    assert types(us, "SSN 666-45-6789") == []  # never issued
    assert types(us, "My SSN is 123456789") == ["ssn_us"]
    assert types(us, "Order 123456789 shipped") == []  # plain digits need the context word
    assert types(PiiDetector(types=["ssn_us"]), "SSN 123-45-6789") == ["ssn_us"]  # named: on whatever the locale
    assert types(PiiDetector(locales=("intl",)), "Driver's license issued 2019-05-01") == []
    with pytest.raises(ValueError):
        PiiDetector(locales=("ca", "mars"))
    with pytest.raises(ValueError):
        PiiDetector(profiles=("medical",))
    with PiiSession(PiiDetector(locales=("fr", "intl"))) as session:
        session.scan("input", "Mon téléphone : 01 23 45 67 89")
        assert session.scan("output", "Je vous rappelle au +33 1 23 45 67 89.").leaks == []  # same number


# ---- M16: sensitive financial identifiers


def test_financial_identifiers():
    finance = PiiDetector(locales=("ca", "us"), profiles=("finance",))
    cases = {
        "IBAN FR76 3000 6000 0112 3456 7890 189": ["iban"],
        "Virement sur GB82WEST12345698765432 svp": ["iban"],
        "Code SWIFT : BNPAFRPPXXX": ["bic"],
        "Numéro de compte : 1234567": ["bank_account"],
        "Chèque 12345-003-1234567": ["bank_account"],
        "Transit 12345, institution 003": ["bank_routing", "bank_routing"],
        "Numéro de transit et d'institution 12345-003": ["bank_routing"],
        "Routing number 021000021": ["bank_routing"],
        "CVV : 123": ["card_security_code"],
    }
    for text, expected in cases.items():
        assert types(finance, text) == expected, text
        assert not set(types(PiiDetector(), text)) & set(FINANCE_TYPES), text  # opt-in
    assert types(finance, "IBAN FR76 3000 6000 0112 3456 7890 188") == []  # mod 97 fails
    us = PiiDetector(locales=("us",), profiles=("finance",))
    assert types(us, "Routing number 021000022") == []  # ABA checksum fails
    assert types(finance, "Code SWIFT : BNPAQQPPXXX") == []  # no such country
    assert types(finance, "La réunion compte 12345 participants et dure 90 minutes, institution reconnue.") == []
    assert types(finance, "Le code est 123.") == []  # a CVV needs its name right before
    assert finance.mask("IBAN FR76 3000 6000 0112 3456 7890 189") == "IBAN [IBAN]"


# ---- M13 (rules side): street addresses


def test_street_addresses():
    address = PiiDetector(profiles=("address",))
    for text, value in (
        ("J'habite au 1234 rue des Érables, Montréal", "1234 rue des Érables"),
        ("Écrivez au 1234, boul. Saint-Laurent Ouest, app. 3", "1234, boul. Saint-Laurent Ouest, app. 3"),
        ("Livrez au 12 rue de la Paix, Paris", "12 rue de la Paix"),
        ("Domicile : 45 avenue d'Italie", "45 avenue d'Italie"),
        ("I live at 55 Maple Street, Toronto ON", "55 Maple Street"),
        ("Ship to 1600 Pennsylvania Avenue NW, Washington", "1600 Pennsylvania Avenue NW"),
        ("221B Baker St. is famous", "221B Baker St."),
        ("Office at 100 King St W, Suite 200", "100 King St W, Suite 200"),
    ):
        findings = address.scan(text)
        assert [(f.type, text[f.start : f.end]) for f in findings] == [("street_address", value)], text
    for text in (
        "La réunion compte 12 participants et dure 90 minutes.",
        "Nous avons 3 places disponibles.",
        "Version 2.3.4567 publiée le 12/03/2025.",
        "Total: 1,250,000 units in 2025.",
        "Ouvert de 9 h à 17 h, 7 jours sur 7.",
        "Le vol AC870 part à 21 h 05.",
    ):
        assert types(address, text) == [], text
    assert types(PiiDetector(), "J'habite au 1234 rue des Érables") == []  # opt-in
    assert address.mask("J'habite au 1234 rue des Érables.") == "J'habite au [ADDRESS]."
    with PiiSession(address) as session:
        session.scan("input", "J'habite au 1234 rue des Érables")
        assert session.scan("output", "Je confirme : 1234, rue des Érables.").leaks == []


def test_policy_passes_locales_and_profiles():
    frame = pd.DataFrame({"note": ["SSN 123-45-6789", "IBAN FR76 3000 6000 0112 3456 7890 189"]})
    masked = PiiPolicy(locales=["ca", "us"], profiles=["finance"]).apply(frame)
    assert masked["note"].tolist() == ["SSN [SSN]", "IBAN [IBAN]"]
    assert PiiPolicy().apply(frame)["pii_count"].tolist() == [0, 0]


def test_words_proper_to_a_type_count_only_right_before_the_value():
    """Review: in the 40-character window they hid SINs and phones that main found (PiiPolicy, inputs)"""
    detector = PiiDetector()
    assert types(detector, f"Employee Jean Tremblay {SIN}") == ["sin"]  # employers hold SINs
    assert types(detector, f"Matricule de l'employé : {SIN}") == ["sin"]
    assert types(detector, "Le membre Jean Tremblay, 514 555 0199") == ["phone"]
    assert types(detector, f"Le client a confirmé, réponse {SIN}") == ["sin"]
    for scope in ("window", "adjacent"):  # still lowered when adjacent (M18)
        for text in (f"No client {SIN}", f"Customer no. {SIN}", f"Code de confirmation : {SIN.replace(' ', '')}"):
            assert detector.scan(text, business_scope=scope) == [], (scope, text)


def test_email_scan_is_linear_on_long_runs_without_at_sign():
    import time

    detector = PiiDetector(types=["email"])
    for text in ("a" * 40_000, "a." * 20_000, "a@" + "b" * 40_000, "a@" + "b." * 20_000):
        start = time.perf_counter()
        detector.scan(text)
        assert time.perf_counter() - start < 0.5  # about 2.4 s on main for 40,000 characters
    assert types(detector, "écrivez à lea@exemple.ca.") == ["email"]
    assert types(detector, "(x_lea.t+1@exemple.qc.ca)") == ["email"]
