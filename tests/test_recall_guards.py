from rainb.recall import grounded, wrong_person

NOTES = ["Aisha's birthday is 9 December.", "My sister Aisha lives in Penang and is a vet.",
         "My manager at Trontal is Khai Wei.", "Caleb is my friend; his birthday is 30 October."]


def test_other_relative_is_rejected_by_following_the_name():
    assert wrong_person("my brother's birthday is on uh", NOTES[0], NOTES)


def test_same_relation_is_accepted():
    assert not wrong_person("my sister's birthday is on uh", NOTES[0], NOTES)
    assert not wrong_person("my manager at Trontal, uh what's his name", NOTES[2], NOTES)


def test_no_relation_said_is_never_blocked():
    assert not wrong_person("Caleb's birthday is on uh", NOTES[3], NOTES)


def test_different_kind_of_place_is_rejected():
    from rainb.recall import wrong_kind

    note = "The Japanese restaurant near work is Uokatsu, on Jalan Telawi."
    assert wrong_kind("Let's go to the fish and chip shop near work. Uh.", note)
    assert not wrong_kind("Let's go to the Japanese place near work. The place uh", note)
    assert not wrong_kind("what is the restaurant near my workplace?", note)
    assert not wrong_kind("I'm going bouldering tonight at uh", "The bouldering gym I go to is Camp5.")


def test_grounding():
    assert grounded("Khai Wei", NOTES[2])
    assert not grounded("Khai Wen", NOTES[2])
