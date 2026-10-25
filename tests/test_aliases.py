from rainb.aliases import Aliases, skeleton

NOTES = ("My manager at Trontal is Khai Wei. The Japanese restaurant near work is Uokatsu, on Jalan Telawi. "
         "The bouldering gym I go to is Camp5. The textbook is Genki. Redis is used for alerts.")
NAMES = ["Trontal", "Uokatsu", "Jalan Telawi", "Camp5", "Khai Wei", "Genki", "Redis"]


def test_skeleton():
    assert skeleton("Toronto") == "trnt"
    assert skeleton("Trontal") == "trntl"


def test_misheard_names_are_fixed():
    a = Aliases(NAMES, NOTES)
    assert a.fix("Toronto?") == "Trontal?"
    assert a.fix("Trontel") == "Trontal"
    assert a.fix("Yuakatsu") == "Uokatsu"


def test_real_words_are_left_alone():
    a = Aliases(NAMES, NOTES)
    for w in ["control", "council", "manager", "camping", "restaurant", "Japanese", "alerts", "the"]:
        assert a.fix(w) is None, w


def test_words_in_the_notes_are_never_changed():
    a = Aliases(["Trontal"], "I flew to Toronto. My manager at Trontal is Khai Wei.")
    assert a.fix("Toronto") is None
