"""Fixed, persona-safe bilingual messages.

Used outside the LLM path (media, errors) and as the deterministic fallback for every agent action
when the LLM is slow, unavailable, or its draft breaks a persona rule. Arabic is Khaleeji.
Persona rules (spec §3): never say AI/bot/assistant; brief, polite, consultative.
"""

UNSUPPORTED_MEDIA = {
    "en": "Thanks for that! Could you type your message here for me? That way I can help you faster 🙏",
    "ar": "مشكور! ممكن تكتب لي رسالتك هنا؟ عشان أقدر أخدمك أسرع 🙏",
}

TEMPORARY_ERROR = {
    "en": "Sorry, give me just a moment, I'll get right back to you.",
    "ar": "المعذرة، عطني لحظة وأرجع لك على طول.",
}

GREETING = {
    "en": "Hello{name}! Thanks for reaching out 😊",
    "ar": "هلا والله{name}! حياك الله 😊",
}

ASK = {
    "target_country": {
        "en": "Which market are you looking at: Dubai/UAE, Saudi Arabia, Qatar or Kuwait?",
        "ar": "أي سوق يهمك: دبي والإمارات، السعودية، قطر ولا الكويت؟",
    },
    "intent": {
        "en": "Is this for investment, for your own use, or for a Golden Visa?",
        "ar": "الشراء للاستثمار، للسكن الشخصي، ولا عشان الإقامة الذهبية؟",
    },
    "budget_usd": {
        "en": "What budget range do you have in mind?",
        "ar": "كم الميزانية اللي في بالك تقريباً؟",
    },
    "timeline": {
        "en": "And when are you planning to buy: right away, within the next 3 months, or are you open to off-plan?",
        "ar": "ومتى ناوي تشتري: على طول، خلال ٣ شهور، ولا تفكر في مشاريع على الخارطة (أوف بلان)؟",
    },
}

OFFER_SLOTS = {
    "en": "Great, let's set up a quick video call with our senior consultant. Which time suits you best?",
    "ar": "ممتاز! خلنا نرتب لك مكالمة فيديو قصيرة مع مستشارنا. أي وقت يناسبك؟",
}
SLOT_LIST_FOOTER = {
    "en": "Just reply with the number that works for you.",
    "ar": "رد علي برقم الموعد اللي يناسبك.",
}
NO_SLOTS = {
    "en": "Perfect, I have everything I need. Our senior consultant will contact you shortly to arrange a call at a time that suits you.",
    "ar": "تمام، كل المعلومات وصلتني. مستشارنا بيتواصل معك قريب عشان نرتب مكالمة بالوقت اللي يناسبك.",
}
SLOT_UNCLEAR = {
    "en": "Which of these times works best for you?",
    "ar": "أي موعد من هذي يناسبك؟",
}
SLOT_TAKEN = {
    "en": "Sorry, that time was just taken. Here are the next available times:",
    "ar": "المعذرة، هذا الموعد انحجز توه. هذي المواعيد المتاحة الحين:",
}
BOOKING_FAILED = {
    "en": "I couldn't lock that time in just now, but our consultant will confirm a time with you shortly.",
    "ar": "ما قدرت أثبت الموعد الحين، بس مستشارنا بيتواصل معك قريب ويأكد معك الوقت.",
}
BOOKED = {
    "en": "Done ✅ Your call is booked. What's your email? I'll send you the calendar invite.",
    "ar": "تم ✅ موعدك محجوز. عطني إيميلك عشان أرسل لك دعوة التقويم.",
}
BOOKING_DETAILS = {
    "en": "📅 {slot}",
    "ar": "📅 {slot}",
}
MEET_LINK = {
    "en": "🎥 Video call: {link}",
    "ar": "🎥 رابط مكالمة الفيديو: {link}",
}
EMAIL_SAVED = {
    "en": "Perfect, the invite is on its way to {email}. Talk soon!",
    "ar": "تمام، أرسلت الدعوة على {email}. نشوفك قريب!",
}
EMAIL_NOTED = {
    "en": "Thanks! I've noted {email}, and our consultant will make sure the invite reaches you.",
    "ar": "مشكور! سجلت إيميلك {email}، ومستشارنا بيتأكد إن الدعوة توصلك.",
}
ALREADY_BOOKED = {
    "en": "You're all set for our call. Anything else I can help you with before then?",
    "ar": "موعدك ثابت إن شاء الله. تبي أساعدك بشي ثاني قبل المكالمة؟",
}
RESCHEDULE = {
    "en": "No problem at all. I'll let our consultant know and they'll reach out to arrange a new time.",
    "ar": "ولا يهمك، ببلغ مستشارنا وبيتواصل معك عشان نرتب وقت ثاني.",
}
# Used when the knowledge base has no verified answer: never guess, confirm on the call instead.
DEFER_QUESTION = {
    "en": "Good question! I want to give you exact, up-to-date details, so I'll confirm that for you on the call.",
    "ar": "سؤال في محله! أبي أعطيك معلومة دقيقة ومحدثة، عشان كذا بأكد لك هالنقطة في المكالمة.",
}
ACK = {
    "en": "Got it, thank you!",
    "ar": "تمام، مشكور!",
}


def pick(template: dict[str, str], language: str | None) -> str:
    return template["ar"] if language == "ar" else template["en"]
