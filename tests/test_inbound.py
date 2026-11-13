from services.inbound import parse_incoming
from tests.conftest import TEST_PHONE_NUMBER_ID, make_text_payload


def test_parses_text_message_with_name_and_language():
    [msg] = parse_incoming(make_text_payload("السلام عليكم", wamid="wamid.A"), 1.0, TEST_PHONE_NUMBER_ID)
    assert (msg.wamid, msg.phone, msg.profile_name) == ("wamid.A", "971501234567", "Ahmed")
    assert msg.text == "السلام عليكم"
    assert msg.language == "ar"
    assert msg.received_at == 1.0


def test_status_callbacks_are_ignored():
    payload = make_text_payload("x")
    value = payload["entry"][0]["changes"][0]["value"]
    del value["messages"]
    value["statuses"] = [{"id": "wamid.X", "status": "delivered", "recipient_id": "971501234567"}]
    assert parse_incoming(payload, 0.0) == []


def test_non_whatsapp_object_ignored():
    assert parse_incoming({"object": "page", "entry": []}, 0.0) == []


def test_other_phone_number_id_ignored():
    assert parse_incoming(make_text_payload("hi"), 0.0, phone_number_id="someone-else") == []


def test_voice_note_has_no_text():
    [msg] = parse_incoming(make_text_payload("", msg_type="audio"), 0.0)
    assert msg.msg_type == "audio" and msg.text is None and msg.language is None


def test_interactive_button_reply_uses_title():
    payload = make_text_payload("x")
    payload["entry"][0]["changes"][0]["value"]["messages"][0] = {
        "from": "971501234567",
        "id": "wamid.B",
        "type": "interactive",
        "interactive": {"type": "button_reply", "button_reply": {"id": "slot_1", "title": "Tomorrow 4 PM"}},
    }
    [msg] = parse_incoming(payload, 0.0)
    assert msg.text == "Tomorrow 4 PM" and msg.language == "en"


def test_reactions_are_ignored():
    payload = make_text_payload("x")
    payload["entry"][0]["changes"][0]["value"]["messages"][0] = {
        "from": "971501234567",
        "id": "wamid.C",
        "type": "reaction",
        "reaction": {"message_id": "wamid.out", "emoji": "👍"},
    }
    assert parse_incoming(payload, 0.0) == []
