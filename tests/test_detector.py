from rainb.detector import COOLDOWN_S, HesitationDetector
from rainb.events import Cancel, StuckCandidate, WordEvent


def say(det, text, start=0.0, step=0.3, speaker=0):
    out, t = [], start
    for word in text.split():
        out += det.feed(WordEvent(word, t, t + step - 0.1, speaker))
        t += step
    return out, t - 0.1  # time the last word ended


def run_pause(det, from_t, seconds, dt=0.1):
    out, t = [], from_t
    while t < from_t + seconds:
        t += dt
        out += det.tick(t)
    return out


def test_pause_after_unfinished_word_fires():
    det = HesitationDetector()
    _, end = say(det, "we should use that paper the one with the")
    fired = run_pause(det, end, 1.0)
    assert len(fired) == 1 and fired[0].reason == "structural"
    assert fired[0].context.endswith("with the")
    assert 0.6 <= fired[0].at - end <= 0.75


def test_pause_after_finished_sentence_is_just_a_breath():
    det = HesitationDetector()
    _, end = say(det, "the paper was great.")
    assert run_pause(det, end, 3.0) == []


def test_short_pause_does_not_fire():
    det = HesitationDetector()
    _, end = say(det, "let's meet at the")
    assert run_pause(det, end, 0.5) == []


def test_trailing_filler_counts_and_lowers_threshold():
    det = HesitationDetector()
    _, end = say(det, "let's go to the japanese place near work. the place uh")
    fired = run_pause(det, end, 1.0)
    assert len(fired) == 1
    assert "filler" in fired[0].signals
    assert fired[0].at - end < 0.6


def test_drawn_out_filler_spellings_count_as_fillers():
    det = HesitationDetector()
    _, end = say(det, "the place... Uhhh...")
    fired = run_pause(det, end, 1.0)
    assert len(fired) == 1 and "filler" in fired[0].signals


def test_direct_question_fires():
    det = HesitationDetector()
    out, _ = say(det, "What is the Japanese restaurant near my workplace?")
    assert [c.reason for c in out if isinstance(c, StuckCandidate)] == ["question"]


def test_direct_question_split_into_fragments_fires_once():
    det = HesitationDetector()
    out, _ = say(det, "What? is the Japanese restaurant? near my workplace?")
    assert [c.reason for c in out if isinstance(c, StuckCandidate)] == ["question"]


def test_question_after_a_silent_early_stall_still_fires():
    det = HesitationDetector()
    _, end = say(det, "wait. where is")
    [early] = run_pause(det, end, 1.0)  # fires before the subject arrives
    det.found_nothing(early.id)  # recall found nothing for "where is"
    out, _ = say(det, "the Japanese restaurant near my workplace?", start=end + 1.2)
    assert [c.reason for c in out if isinstance(c, StuckCandidate)] == ["question"]


def test_one_lookup_per_pause_even_when_it_finds_nothing():
    det = HesitationDetector()
    _, end = say(det, "where is")
    [c] = run_pause(det, end, 0.8)
    det.found_nothing(c.id)
    assert run_pause(det, end + 0.8, 3.0) == []


def test_small_talk_questions_do_not_fire():
    det = HesitationDetector()
    out, _ = say(det, "How are you doing today? What?")
    assert [c for c in out if isinstance(c, StuckCandidate)] == []


def test_explicit_phrase_fires_without_pause():
    det = HesitationDetector()
    out, _ = say(det, "uh what's the place called again?")
    assert [c.reason for c in out if isinstance(c, StuckCandidate)] == ["explicit"]


def test_other_speaker_cannot_trigger_and_handover_suppresses():
    det = HesitationDetector()
    out, end = say(det, "what's it called", speaker=1)
    assert out == [] and run_pause(det, end, 2.0) == []
    _, end = say(det, "i went to the", start=end + 0.2)
    det.feed(WordEvent("yeah", end + 0.2, end + 0.4, speaker=1))
    assert run_pause(det, end, 2.0) == []


def test_resuming_speech_cancels_and_cooldown_blocks_repeat():
    det = HesitationDetector()
    _, end = say(det, "the paper by")
    [c] = run_pause(det, end, 1.0)
    out = det.feed(WordEvent("uh", end + 1.1, end + 1.3))
    assert out == []  # a filler is not recovery
    out = det.feed(WordEvent("a", end + 1.3, end + 1.35))
    assert out == []  # nor is a function word (often a misheard "uh")
    out = det.feed(WordEvent("vaswani", end + 1.4, end + 1.8))
    assert out == [Cancel(c.id, "self_recovery")]
    _, end2 = say(det, "and the", start=end + 2.0)
    assert run_pause(det, end2, 2.0) == []  # grace period after recovering
    _, end3 = say(det, "and then the", start=end + 6.0)
    assert len(run_pause(det, end3, 1.0)) == 1  # recovery cleared the 20 s cool-down
    _, end4 = say(det, "so the", start=end3 + 2.0)
    assert run_pause(det, end4, 2.0) == []  # that one wasn't recovered: full cool-down


def test_explicit_phrase_right_after_recovery_still_fires():
    det = HesitationDetector()
    _, end = say(det, "the place near the")
    [c] = run_pause(det, end, 1.0)
    out, _ = say(det, "uh what's it called", start=end + 1.2)
    kinds = [type(x).__name__ for x in out]
    assert kinds == ["Cancel", "StuckCandidate"]
    assert out[1].reason == "explicit"


def test_explicit_fires_once_not_on_every_following_word():
    det = HesitationDetector()
    out, _ = say(det, "what's it called")
    out2, _ = say(det, "the thing", start=30.0)
    assert len([x for x in out + out2 if isinstance(x, StuckCandidate)]) == 1


def test_slow_speaker_gets_a_higher_threshold():
    det = HesitationDetector()
    # habitual 0.5 s gaps with some spread
    t = 0.0
    for i in range(20):
        det.feed(WordEvent(f"w{i}", t, t + 0.2))
        t += 0.2 + (0.35 if i % 2 else 0.55)
    det.feed(WordEvent("with", t, t + 0.2))
    assert det.threshold(t + 0.2) > 0.6


def test_manual_always_fires():
    det = HesitationDetector()
    c = det.manual(5.0)
    assert c.reason == "manual"
