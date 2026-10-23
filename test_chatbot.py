"""
Tests for the chatbot. Run with:  python -m pytest test_chatbot.py
(These tests do not need TensorFlow or the trained model.)
"""
from chatbot import get_reply, pretty


def test_answers_about_a_named_disease():
    r = get_reply("What is eczema?")
    assert r["disease"] == "Eczema"
    assert "Eczema" in r["reply"]


def test_uses_latest_prediction_when_no_disease_named():
    r = get_reply("how is it treated", disease="Tinea")
    assert r["disease"] == "Tinea"
    assert "managed" in r["reply"]


def test_understands_aliases():
    assert get_reply("symptoms of ringworm")["disease"] == "Tinea"
    assert get_reply("I have pimples")["disease"] == "Acne"


def test_contagious_question():
    assert "not contagious" in get_reply("is psoriasis contagious")["reply"]


def test_urgent_message_takes_priority():
    r = get_reply("I can't breathe and my face is swollen")
    assert "immediately" in r["reply"]


def test_lists_diseases():
    r = get_reply("which diseases can you detect", class_names=["Acne", "Eczema"])
    assert "2 skin conditions" in r["reply"]


def test_unknown_question_gets_help():
    assert "didn't understand" in get_reply("asdfgh")["reply"]


def test_readable_names():
    assert pretty("DrugEruption") == "Drug Eruption"
    assert pretty("Seborrh_Keratoses") == "Seborrheic Keratoses"
