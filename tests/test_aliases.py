from rainb.aliases import Aliases, skeleton

NOTES = ["My manager at Trontal is Khai Wei.",
         "The Trontal alert system has 60000 alerts and uses Redis.",
         "The Japanese restaurant near work is Uokatsu, on Jalan Telawi.",
         "The bouldering gym I go to is Camp5.", "The textbook is Genki."]
NAMES = ["Trontal", "Uokatsu", "Jalan Telawi", "Camp5", "Khai Wei", "Genki", "Redis"]
KNOWN = frozenset({"toronto", "control", "council", "manager", "camping", "camp", "restaurant",
                   "japanese", "alerts", "the", "flew", "last", "year"})


def say(a, sentence):
    out, seen = [], []
    for w in sentence.split():
        out.append(a.fix(w, seen[-8:]) or w)
        seen.append(w)
    return " ".join(out)


def test_skeleton():
    assert skeleton("Toronto") == "trnt"
    assert skeleton("Trontal") == "trntl"


def test_non_words_that_sound_like_a_name_are_fixed():
    a = Aliases(NAMES, NOTES, KNOWN)
    assert a.fix("Trontel") == "Trontal"
    assert a.fix("Yuakatsu") == "Uokatsu"
    assert a.fix("Readies") == "Redis"


def test_real_toronto_is_fixed_only_with_supporting_context():
    a = Aliases(NAMES, NOTES, KNOWN)
    assert say(a, "Who was my manager in Toronto?") == "Who was my manager in Trontal?"
    assert say(a, "the Toronto alert system is down") == "the Toronto alert system is down"  # context comes after
    assert say(a, "I flew to Toronto last year") == "I flew to Toronto last year"


def test_real_words_are_left_alone():
    a = Aliases(NAMES, NOTES, KNOWN)
    for w in ["control", "council", "manager", "camping", "camp", "restaurant", "Japanese", "the"]:
        assert a.fix(w, ["my", "manager", "alert"]) is None, w


def test_words_in_the_notes_are_never_changed():
    a = Aliases(["Trontal"], ["I flew to Toronto.", "My manager at Trontal is Khai Wei."], KNOWN)
    assert a.fix("Toronto", ["manager"]) is None
