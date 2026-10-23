"""
Healthcare chatbot for the skin disease project.

A rule-based (knowledge-base) chatbot. It works offline and needs no API key.
It understands:
  - which disease the user is asking about (by name or common alias)
  - what they want to know: overview, symptoms, causes, treatment,
    prevention, whether it is contagious, or when to see a doctor
  - general questions: greetings, help, list of diseases, how the model works

If the user does not name a disease, the chatbot uses the disease from the
latest prediction (sent by the website as "disease").

IMPORTANT: answers are general health information for education only,
not a medical diagnosis or treatment plan.
"""

import re

DISCLAIMER = (
    "This is general information for education only, not a diagnosis. "
    "Please consult a qualified dermatologist for proper assessment."
)

URGENT_REPLY = (
    "Some of what you describe can be serious. Please seek medical care "
    "immediately (visit an emergency department or call your local emergency "
    "number, 112 in India) if you have:\n"
    "• difficulty breathing or swelling of the face, lips or tongue\n"
    "• a rapidly spreading rash with fever or feeling very unwell\n"
    "• widespread blistering or peeling skin, or sores in the mouth or eyes\n"
    "• a rash after starting a new medicine together with any of the above"
)

URGENT_WORDS = [
    "cant breathe", "can't breathe", "cannot breathe", "difficulty breathing",
    "trouble breathing", "short of breath", "swollen face", "swelling face",
    "lips swelling", "tongue swelling", "throat swelling", "anaphylaxis",
    "skin peeling off", "skin is peeling", "blisters all over", "high fever",
    "emergency", "fainted", "unconscious",
]

# ---------------------------------------------------------------------------
# Knowledge base: one entry per class. "aliases" are words users may type.
# ---------------------------------------------------------------------------
KB = {
    "Acne": {
        "aliases": ["acne", "pimple", "pimples", "zit", "zits", "blackhead", "whitehead", "breakout"],
        "overview": "Acne is a common skin condition where hair follicles become blocked with oil and dead skin cells. It mostly affects the face, chest and back, and is very common in teenagers and young adults.",
        "symptoms": ["Blackheads and whiteheads", "Red, tender bumps (papules) or pus-filled pimples", "Painful lumps under the skin (nodules or cysts) in severe cases", "Possible dark marks or scars after healing"],
        "causes": ["Excess oil (sebum) production", "Blocked pores from dead skin cells", "Bacteria on the skin", "Hormonal changes (puberty, periods), some medicines, and heavy oily cosmetics"],
        "treatment": ["Gentle cleansing twice a day", "Over-the-counter products containing benzoyl peroxide or salicylic acid", "A dermatologist may prescribe topical retinoids, antibiotics or other medicines for moderate to severe acne"],
        "prevention": ["Wash the face gently; avoid harsh scrubbing", "Do not squeeze or pick pimples", "Use non-comedogenic (oil-free) skincare and makeup", "Remove makeup before sleeping"],
        "contagious": "No, acne is not contagious.",
        "see_doctor": "See a dermatologist if acne is painful, leaves scars, does not improve after a few weeks of over-the-counter care, or affects your confidence.",
    },
    "Actinic_Keratosis": {
        "aliases": ["actinic keratosis", "actinic", "solar keratosis", "ak"],
        "overview": "Actinic keratosis is a rough, scaly patch on skin that has had many years of sun exposure. It is considered precancerous because a small number can turn into squamous cell skin cancer.",
        "symptoms": ["Rough, dry or scaly patch, often smaller than 2.5 cm", "Pink, red or brown colour", "Usually on the face, ears, scalp, hands or forearms", "May itch, burn or feel like sandpaper"],
        "causes": ["Long-term exposure to ultraviolet (UV) light from the sun or tanning beds", "More common in older adults and people with fair skin"],
        "treatment": ["A doctor can remove it by freezing (cryotherapy)", "Prescription creams or gels", "Other procedures such as photodynamic therapy or minor surgery"],
        "prevention": ["Use broad-spectrum sunscreen (SPF 30 or more) daily", "Wear hats and protective clothing", "Avoid the sun between 10 am and 4 pm", "Avoid tanning beds"],
        "contagious": "No, actinic keratosis is not contagious.",
        "see_doctor": "Get any rough, scaly patch checked by a doctor, especially if it grows, bleeds, becomes painful or thick, since it can occasionally become skin cancer.",
    },
    "Benign_tumors": {
        "aliases": ["benign tumor", "benign tumour", "benign tumors", "benign", "lipoma", "skin tag", "dermatofibroma", "cyst", "skin growth", "lump"],
        "overview": "Benign skin tumours are non-cancerous growths on or under the skin, such as skin tags, lipomas, dermatofibromas and cysts. They usually grow slowly and do not spread.",
        "symptoms": ["A lump, bump or growth that is usually painless", "Slow growth over months or years", "Usually smooth with a regular shape"],
        "causes": ["Overgrowth of normal skin, fat or fibrous cells", "Genetics, friction and age can play a role"],
        "treatment": ["Often no treatment is needed", "A doctor can remove it if it is painful, irritated or for cosmetic reasons"],
        "prevention": ["Most benign growths cannot be prevented", "Regularly check your skin for new or changing growths"],
        "contagious": "No, benign skin tumours are not contagious.",
        "see_doctor": "See a doctor if a growth changes quickly in size, shape or colour, bleeds, becomes painful, or if you are unsure what it is.",
    },
    "Bullous": {
        "aliases": ["bullous", "bullous disease", "pemphigoid", "pemphigus", "blister disease", "blistering"],
        "overview": "Bullous diseases are a group of conditions that cause large fluid-filled blisters (bullae) on the skin. Many, such as bullous pemphigoid and pemphigus, are autoimmune, meaning the immune system attacks the skin.",
        "symptoms": ["Large, tense or fragile blisters", "Itching, sometimes before blisters appear", "Raw areas when blisters break", "Blisters in the mouth in some types"],
        "causes": ["Autoimmune reactions against proteins that hold skin layers together", "Some cases are triggered by certain medicines", "Genetic forms exist but are rare"],
        "treatment": ["Needs diagnosis and treatment by a dermatologist", "Treatment often includes medicines that calm the immune system and careful wound care"],
        "prevention": ["Most types cannot be prevented", "Avoid medicines that triggered blisters before", "Protect fragile skin from injury"],
        "contagious": "No, autoimmune bullous diseases are not contagious.",
        "see_doctor": "See a doctor promptly for unexplained blisters. Seek urgent care if blisters are widespread, in the mouth or eyes, or come with fever.",
    },
    "Candidiasis": {
        "aliases": ["candidiasis", "candida", "yeast infection", "thrush", "fungal yeast"],
        "overview": "Candidiasis is an infection caused by Candida yeast. On the skin it usually appears in warm, moist folds such as the armpits, groin, under the breasts or between fingers.",
        "symptoms": ["Red, moist, itchy rash", "Small red bumps or pustules near the edge (satellite spots)", "Burning or soreness", "White patches in the mouth (oral thrush)"],
        "causes": ["Overgrowth of yeast in warm, moist areas", "Diabetes, obesity, antibiotics, tight clothing and a weak immune system increase risk"],
        "treatment": ["Antifungal creams or powders", "A doctor may prescribe oral antifungal medicine for severe or repeated infections", "Keep the area clean and dry"],
        "prevention": ["Keep skin folds clean and dry", "Wear loose, breathable cotton clothing", "Change out of wet clothes quickly", "Keep blood sugar controlled if diabetic"],
        "contagious": "It can sometimes spread through close skin contact, but it mostly develops from yeast already on the body.",
        "see_doctor": "See a doctor if it does not improve after about a week of treatment, keeps coming back, or if you have diabetes or a weak immune system.",
    },
    "DrugEruption": {
        "aliases": ["drug eruption", "drugeruption", "drug rash", "drug reaction", "medicine rash", "medication rash", "allergic reaction to medicine"],
        "overview": "A drug eruption is a skin reaction to a medicine. Most are mild rashes, but a few rare reactions can be serious and need emergency care.",
        "symptoms": ["Red rash or spots, often on the trunk, appearing days to weeks after starting a medicine", "Itching or hives", "Rarely: blisters, peeling skin, mouth sores, fever or swelling (serious warning signs)"],
        "causes": ["An allergic or immune reaction to a medicine", "Common triggers include some antibiotics, anti-seizure medicines and pain relievers"],
        "treatment": ["A doctor will decide whether to stop the medicine; do not stop prescribed medicine on your own without advice", "Antihistamines or creams may be used for itching", "Serious reactions need hospital care"],
        "prevention": ["Tell every doctor and pharmacist about past drug reactions", "Keep a written list of medicines that caused reactions"],
        "contagious": "No, drug eruptions are not contagious.",
        "see_doctor": "Contact a doctor soon after a new rash appears with a new medicine. Seek emergency care for blisters, peeling skin, mouth or eye sores, fever, or breathing difficulty.",
    },
    "Eczema": {
        "aliases": ["eczema", "atopic dermatitis", "dermatitis", "dry itchy skin", "itchy skin"],
        "overview": "Eczema (atopic dermatitis) is a long-term condition that makes skin dry, itchy and inflamed. It often starts in childhood and comes and goes in flares.",
        "symptoms": ["Dry, itchy skin, often worse at night", "Red or darker patches, commonly in elbow and knee creases, hands, neck and face", "Thickened skin from scratching", "Oozing or crusting during flares"],
        "causes": ["A weakened skin barrier and an overactive immune response", "Family history of eczema, asthma or allergies", "Triggers such as soaps, sweat, dust, stress, weather changes and certain fabrics"],
        "treatment": ["Moisturise often, especially right after bathing", "Avoid known triggers", "A doctor may prescribe anti-inflammatory creams for flares"],
        "prevention": ["Use fragrance-free moisturisers and mild soaps", "Take short lukewarm baths", "Wear soft cotton clothing", "Keep nails short to reduce damage from scratching"],
        "contagious": "No, eczema is not contagious.",
        "see_doctor": "See a doctor if itching disturbs sleep, the rash spreads, or there are signs of infection such as yellow crusts, pus or fever.",
    },
    "Infestations_Bites": {
        "aliases": ["infestation", "infestations", "bites", "insect bite", "bug bite", "scabies", "lice", "mosquito bite", "bed bugs", "bedbug", "mites"],
        "overview": "This group includes skin problems caused by insects and parasites, such as insect bites, bed bugs, lice and scabies mites.",
        "symptoms": ["Itchy red bumps or welts", "Scabies: intense itching, worse at night, with tiny lines or bumps between fingers, wrists, waist", "Lice: itching of the scalp with eggs (nits) on hair", "Bites often appear in groups or lines"],
        "causes": ["Bites from mosquitoes, fleas, bed bugs or other insects", "Scabies mites burrowing into the skin", "Head lice living on the scalp"],
        "treatment": ["Most insect bites settle with cool compresses and anti-itch creams", "Scabies and lice need specific treatments prescribed or recommended by a doctor or pharmacist, often for the whole household", "Wash bedding and clothes in hot water"],
        "prevention": ["Use insect repellent and mosquito nets", "Avoid sharing combs, hats, towels and bedding", "Check for bed bugs when travelling"],
        "contagious": "Scabies and lice spread by close contact and shared items. Simple insect bites are not contagious.",
        "see_doctor": "See a doctor if itching is severe or lasts more than a week, several family members are affected, or a bite becomes red, hot, swollen and painful (possible infection).",
    },
    "Lichen": {
        "aliases": ["lichen", "lichen planus", "lichen sclerosus"],
        "overview": "Lichen planus is an inflammatory condition that can affect the skin, mouth, nails and scalp. It is thought to be related to the immune system.",
        "symptoms": ["Small, shiny, flat-topped purple or dark bumps", "Itching", "Commonly on wrists, ankles and lower back", "Lacy white patches inside the mouth"],
        "causes": ["The immune system attacking skin or mucous membrane cells", "Sometimes linked to certain medicines or hepatitis C"],
        "treatment": ["Often clears by itself over months to a couple of years", "A doctor may prescribe creams or other medicines to reduce itching and inflammation"],
        "prevention": ["It usually cannot be prevented", "Avoid scratching and skin injury, which can cause new spots"],
        "contagious": "No, lichen planus is not contagious.",
        "see_doctor": "See a doctor for diagnosis, especially if the mouth, genitals, nails or scalp are affected, or if itching is severe.",
    },
    "Lupus": {
        "aliases": ["lupus", "sle", "butterfly rash", "discoid lupus", "cutaneous lupus"],
        "overview": "Lupus is an autoimmune disease in which the immune system attacks healthy tissue. Cutaneous lupus affects the skin; systemic lupus can also affect joints, kidneys and other organs.",
        "symptoms": ["Butterfly-shaped rash across the cheeks and nose", "Round, scaly patches (discoid lupus) that may scar", "Rash that worsens after sun exposure", "In systemic lupus: tiredness, joint pain and fever"],
        "causes": ["An autoimmune reaction; exact cause unknown", "Genetics, hormones, sunlight and some medicines can trigger it", "More common in women"],
        "treatment": ["Needs diagnosis and long-term care by a doctor", "Sun protection is important", "Doctors may use creams and medicines that control the immune system"],
        "prevention": ["Lupus cannot be prevented, but flares can be reduced", "Use sunscreen and protective clothing", "Follow medical advice and attend check-ups"],
        "contagious": "No, lupus is not contagious.",
        "see_doctor": "See a doctor if you have a facial rash with joint pain, tiredness or fever, or a rash that worsens in sunlight.",
    },
    "Moles": {
        "aliases": ["mole", "moles", "nevus", "nevi", "beauty spot", "birthmark"],
        "overview": "Moles (nevi) are common growths made of pigment cells. Most are harmless, but a changing mole can be an early sign of melanoma.",
        "symptoms": ["Small brown, black or skin-coloured spot", "Flat or raised, usually round with a smooth edge", "Usually stays the same over time"],
        "causes": ["Clusters of pigment cells (melanocytes)", "Genetics and sun exposure"],
        "treatment": ["Usually no treatment is needed", "A doctor can remove a mole that looks suspicious or causes irritation"],
        "prevention": ["Protect skin from the sun", "Check moles monthly using the ABCDE rule: Asymmetry, irregular Border, uneven Colour, Diameter over 6 mm, Evolving (changing)"],
        "contagious": "No, moles are not contagious.",
        "see_doctor": "See a dermatologist if a mole changes in size, shape or colour, has an irregular border, bleeds, itches or looks different from your other moles.",
    },
    "Psoriasis": {
        "aliases": ["psoriasis", "plaque psoriasis", "scaly patches"],
        "overview": "Psoriasis is a long-term immune-related condition that makes skin cells build up too quickly, forming thick, scaly patches. It comes and goes in flares.",
        "symptoms": ["Raised red or darker patches covered with silvery-white scales", "Common on elbows, knees, scalp and lower back", "Itching or soreness", "Pitted or thickened nails; joint pain in some people (psoriatic arthritis)"],
        "causes": ["An overactive immune system speeding up skin cell growth", "Family history", "Triggers such as stress, infections, skin injury, smoking, alcohol and some medicines"],
        "treatment": ["Moisturisers and medicated creams", "Light therapy (phototherapy)", "Tablets or injections for moderate to severe psoriasis, prescribed by a specialist"],
        "prevention": ["It cannot be cured or fully prevented, but flares can be reduced", "Keep skin moisturised", "Manage stress, avoid smoking and limit alcohol"],
        "contagious": "No, psoriasis is not contagious.",
        "see_doctor": "See a doctor if patches are spreading, painful, affecting daily life, or if you have joint pain or swelling.",
    },
    "Rosacea": {
        "aliases": ["rosacea", "red face", "facial redness", "flushing"],
        "overview": "Rosacea is a long-term condition that causes redness and visible blood vessels on the face, sometimes with small pimple-like bumps. It is most common in adults over 30.",
        "symptoms": ["Persistent redness on cheeks, nose, chin or forehead", "Frequent flushing or blushing", "Visible small blood vessels", "Small red bumps or pus-filled spots", "Burning or stinging; sometimes dry, irritated eyes"],
        "causes": ["Exact cause unknown; blood vessel and immune factors are involved", "Triggers include sun, heat, spicy food, hot drinks, alcohol and stress"],
        "treatment": ["Identify and avoid triggers", "Gentle skincare and daily sunscreen", "A doctor may prescribe creams, tablets or laser treatment"],
        "prevention": ["Keep a diary to find your triggers", "Protect your face from sun and extreme temperatures", "Use mild, fragrance-free products"],
        "contagious": "No, rosacea is not contagious.",
        "see_doctor": "See a doctor if facial redness persists, bumps appear, or your eyes become red and irritated.",
    },
    "Seborrh_Keratoses": {
        "aliases": ["seborrheic keratosis", "seborrheic keratoses", "seborrh keratoses", "seborrhoeic keratosis", "seb k", "age spot growth", "barnacle"],
        "overview": "Seborrheic keratoses are common, harmless skin growths that appear with age. They often look waxy and 'stuck on' to the skin.",
        "symptoms": ["Brown, black or light tan growths", "Waxy, scaly or slightly raised surface", "Look as if they are stuck onto the skin", "Common on chest, back, face and shoulders"],
        "causes": ["Ageing and genetics", "They become more common after age 40"],
        "treatment": ["No treatment is needed", "A doctor can remove them if irritated or for cosmetic reasons"],
        "prevention": ["They cannot be prevented"],
        "contagious": "No, seborrheic keratoses are not contagious.",
        "see_doctor": "See a doctor if a growth changes quickly, bleeds, or you are not sure whether it is a seborrheic keratosis or something else such as melanoma.",
    },
    "SkinCancer": {
        "aliases": ["skin cancer", "skincancer", "melanoma", "basal cell carcinoma", "squamous cell carcinoma", "bcc", "scc", "cancer"],
        "overview": "Skin cancer is the abnormal growth of skin cells, most often caused by UV damage. The main types are basal cell carcinoma, squamous cell carcinoma and melanoma, which is the most dangerous. Early detection greatly improves outcomes.",
        "symptoms": ["A new spot or a mole that changes in size, shape or colour", "A sore that does not heal or keeps bleeding", "A shiny, pearly bump or a firm red lump", "A scaly patch that crusts or bleeds", "Use the ABCDE rule for moles: Asymmetry, Border, Colour, Diameter, Evolving"],
        "causes": ["Ultraviolet radiation from the sun or tanning beds", "Fair skin, many moles, history of sunburns and family history increase risk"],
        "treatment": ["Requires diagnosis by a doctor, usually with a biopsy", "Treatment may include surgical removal and, for advanced cases, other therapies chosen by a specialist"],
        "prevention": ["Use broad-spectrum sunscreen (SPF 30 or more)", "Wear hats, sunglasses and protective clothing", "Avoid tanning beds", "Check your skin regularly for changes"],
        "contagious": "No, skin cancer is not contagious.",
        "see_doctor": "See a doctor as soon as possible for any new, changing, bleeding or non-healing spot. Early diagnosis is very important.",
    },
    "Sun_Sunlight_Damage": {
        "aliases": ["sun damage", "sunlight damage", "sunburn", "sun burn", "photodamage", "sun spots", "sunspot", "tan", "solar damage"],
        "overview": "Sun damage includes changes caused by ultraviolet (UV) exposure, such as sunburn, dark spots (solar lentigines), wrinkles, rough texture and uneven skin tone.",
        "symptoms": ["Red, painful skin after sun exposure (sunburn)", "Brown spots on sun-exposed areas", "Wrinkles, thin or leathery skin", "Rough, dry patches"],
        "causes": ["Repeated or intense UV exposure from sunlight or tanning beds"],
        "treatment": ["For sunburn: cool compresses, moisturiser, drinking fluids and avoiding further sun", "A dermatologist can treat dark spots and other changes with creams or procedures"],
        "prevention": ["Apply broad-spectrum sunscreen (SPF 30 or more) and reapply every 2 hours outdoors", "Wear hats and long sleeves", "Seek shade between 10 am and 4 pm"],
        "contagious": "No, sun damage is not contagious.",
        "see_doctor": "See a doctor for severe or blistering sunburn with fever, or any spot that changes, bleeds or does not heal.",
    },
    "Tinea": {
        "aliases": ["tinea", "ringworm", "fungal infection", "fungus", "athlete's foot", "athletes foot", "jock itch", "tinea corporis", "tinea cruris", "daad"],
        "overview": "Tinea (ringworm) is a common fungal skin infection, not caused by a worm. It can affect the body, groin (jock itch), feet (athlete's foot), scalp and nails.",
        "symptoms": ["Ring-shaped, red, scaly patch with a clearer centre", "Itching", "Patches that slowly spread outward", "Cracked, peeling skin between toes (athlete's foot)"],
        "causes": ["Fungi (dermatophytes) that grow in warm, moist conditions", "Spread by contact with infected people, animals, towels, floors or clothing"],
        "treatment": ["Antifungal creams usually work for small areas when used for the full recommended time", "A doctor may prescribe oral antifungals for scalp, nail or widespread infection", "Avoid using steroid creams on fungal infections without medical advice, as they can make it worse"],
        "prevention": ["Keep skin clean and dry, especially skin folds and feet", "Do not share towels, clothes or combs", "Wear loose cotton clothing and change socks daily", "Wear footwear in public showers"],
        "contagious": "Yes, tinea is contagious through skin contact and shared items.",
        "see_doctor": "See a doctor if it does not improve after 2 weeks of treatment, is spreading, affects the scalp or nails, or keeps coming back.",
    },
    "Unknown_Normal": {
        "aliases": ["normal", "unknown", "healthy skin", "normal skin", "unknown normal"],
        "overview": "This class contains images of normal skin or conditions that do not belong to the other categories. If your result is 'Unknown/Normal', the model did not find a clear match with the trained diseases.",
        "symptoms": ["No specific disease pattern detected by the model"],
        "causes": ["The skin may be healthy, or the condition may not be among the 22 trained classes, or the photo may be unclear"],
        "treatment": ["No treatment is suggested based on this result"],
        "prevention": ["Maintain good skincare: gentle cleansing, moisturising and daily sunscreen"],
        "contagious": "Not applicable.",
        "see_doctor": "If you still have symptoms or concerns, see a doctor. The model can miss conditions, so a normal result does not rule out a problem.",
    },
    "Vascular_Tumors": {
        "aliases": ["vascular tumor", "vascular tumour", "vascular tumors", "hemangioma", "haemangioma", "cherry angioma", "pyogenic granuloma", "port wine stain", "strawberry mark"],
        "overview": "Vascular tumours are growths made of blood vessels. Most, such as cherry angiomas and infant haemangiomas, are harmless.",
        "symptoms": ["Bright red, purple or blue spot or bump", "May bleed easily if injured (for example pyogenic granuloma)", "Infant haemangiomas may grow in the first months and then slowly shrink"],
        "causes": ["Overgrowth of blood vessel cells", "Age, genetics, pregnancy or minor injury can play a role"],
        "treatment": ["Often no treatment is needed", "A doctor may use laser, minor surgery or medicines if it bleeds, grows quickly or affects vision, breathing or feeding in babies"],
        "prevention": ["Most cannot be prevented"],
        "contagious": "No, vascular tumours are not contagious.",
        "see_doctor": "See a doctor if it bleeds repeatedly, grows quickly, or in a baby if it is near the eyes, nose, mouth or nappy area.",
    },
    "Vasculitis": {
        "aliases": ["vasculitis", "inflamed blood vessels", "purpura", "palpable purpura"],
        "overview": "Vasculitis is inflammation of blood vessels. In the skin it often causes red or purple spots that do not fade when pressed, usually on the legs. It can sometimes affect internal organs.",
        "symptoms": ["Red or purple raised spots, often on the lower legs, that do not blanch when pressed", "Hives, ulcers or blisters in some cases", "Possible fever, tiredness, joint pain or abdominal pain"],
        "causes": ["Immune reactions triggered by infections, medicines or autoimmune diseases", "Sometimes no cause is found"],
        "treatment": ["Needs medical assessment to find the cause and check other organs", "Treatment depends on the cause and severity"],
        "prevention": ["It usually cannot be prevented", "Tell doctors about any previous reaction to medicines"],
        "contagious": "No, vasculitis is not contagious.",
        "see_doctor": "See a doctor promptly for purple spots that do not fade when pressed. Seek urgent care if they come with fever, feeling very unwell, abdominal pain or blood in urine.",
    },
    "Vitiligo": {
        "aliases": ["vitiligo", "white patches", "leucoderma", "leukoderma", "depigmentation"],
        "overview": "Vitiligo is a condition in which skin loses its pigment cells, causing white or light patches. It is linked to the immune system and is not harmful to general health.",
        "symptoms": ["Smooth white or light patches of skin", "Often on the face, hands, around the mouth and eyes, or joints", "Early whitening of hair, eyebrows or eyelashes in affected areas"],
        "causes": ["The immune system attacking pigment cells (melanocytes)", "Family history and other autoimmune conditions increase risk"],
        "treatment": ["A dermatologist may suggest creams, light therapy or other treatments to restore colour", "Sunscreen protects white patches from sunburn", "Cosmetic camouflage is an option"],
        "prevention": ["It cannot be prevented", "Protect patches from sunburn"],
        "contagious": "No, vitiligo is not contagious.",
        "see_doctor": "See a dermatologist if you notice new white patches. Early treatment may give better results. Support is also available if it affects your confidence.",
    },
    "Warts": {
        "aliases": ["wart", "warts", "verruca", "verrucae", "plantar wart", "common wart", "molluscum"],
        "overview": "Warts are small, rough growths caused by the human papillomavirus (HPV). They are common in children and young adults and are usually harmless.",
        "symptoms": ["Small, rough, raised bumps", "Often on hands, fingers, knees or feet", "Plantar warts on the soles can be painful when walking", "Tiny black dots inside (small clotted blood vessels)"],
        "causes": ["HPV infection entering through small cuts in the skin"],
        "treatment": ["Many warts disappear on their own within months to years", "Over-the-counter salicylic acid treatments", "A doctor can freeze or remove stubborn warts"],
        "prevention": ["Do not pick or bite warts", "Wear footwear in public showers and pools", "Do not share towels or razors", "Keep cuts covered"],
        "contagious": "Yes, warts can spread through skin contact and shared items.",
        "see_doctor": "See a doctor if warts are painful, spreading, on the face or genitals, bleed, or if you have diabetes or a weak immune system.",
    },
}

INTENTS = {
    "symptoms": ["symptom", "symptoms", "sign", "signs", "look like", "looks like", "appear", "feel like", "how do i know", "identify"],
    "causes": ["cause", "causes", "caused", "why do", "why does", "reason", "reasons", "how do you get", "how did i get", "risk", "trigger", "triggers"],
    "treatment": ["treat", "treatment", "treatments", "cure", "cured", "medicine", "medicines", "medication", "remedy", "remedies", "heal", "get rid", "therapy", "cream", "how to fix"],
    "prevention": ["prevent", "prevention", "avoid", "stop it", "protect", "care tips", "tips", "precaution", "precautions"],
    "contagious": ["contagious", "spread", "spreads", "infectious", "catch it", "transmit", "share"],
    "see_doctor": ["doctor", "dermatologist", "hospital", "serious", "dangerous", "worried", "when should", "consult", "emergency"],
    "overview": ["what is", "what's", "whats", "tell me about", "about", "explain", "meaning", "define", "info", "information"],
}


def _norm(text):
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _key_of(name):
    """'Seborrh_Keratoses' -> 'seborrhkeratoses' for loose comparisons."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


_KB_BY_KEY = {_key_of(k): k for k in KB}


def kb_name_for(class_name):
    """Map a class name from class_names.json to a KB entry (or None)."""
    if not class_name:
        return None
    k = _key_of(class_name)
    if k in _KB_BY_KEY:
        return _KB_BY_KEY[k]
    for kb_name, entry in KB.items():
        if any(_key_of(a) == k for a in entry["aliases"]):
            return kb_name
    return None


DISPLAY_NAMES = {
    "Actinic_Keratosis": "Actinic Keratosis",
    "Benign_tumors": "Benign Tumors",
    "Bullous": "Bullous Disease",
    "DrugEruption": "Drug Eruption",
    "Infestations_Bites": "Infestations and Bites",
    "Lichen": "Lichen Planus",
    "Seborrh_Keratoses": "Seborrheic Keratoses",
    "SkinCancer": "Skin Cancer",
    "Sun_Sunlight_Damage": "Sun / Sunlight Damage",
    "Unknown_Normal": "Unknown / Normal",
    "Vascular_Tumors": "Vascular Tumors",
}


def pretty(name):
    """Readable name for a class, e.g. 'DrugEruption' -> 'Drug Eruption'."""
    kb = kb_name_for(name) or name
    return DISPLAY_NAMES.get(kb, name.replace("_", " "))


def _find_disease(text_norm):
    """Return the KB disease mentioned in the message (longest alias wins)."""
    padded = f" {text_norm} "
    best, best_len = None, 0
    for name, entry in KB.items():
        for alias in entry["aliases"] + [pretty(name).lower()]:
            a = _norm(alias)
            if f" {a} " in padded and len(a) > best_len:
                best, best_len = name, len(a)
    return best


def _find_intent(text_norm):
    padded = f" {text_norm} "
    tokens = text_norm.split()
    for intent in ["symptoms", "causes", "treatment", "prevention", "contagious", "see_doctor", "overview"]:
        for word in INTENTS[intent]:
            w = _norm(word)
            if " " in w:
                if f" {w} " in padded:
                    return intent
            elif w in tokens or (len(w) >= 5 and any(t.startswith(w) for t in tokens)):
                # prefix match handles 'treated', 'causing', 'prevented', 'spreading'
                return intent
    return None


def _bullets(items):
    return "\n".join(f"• {item}" for item in items)


def _answer(disease, intent):
    e = KB[disease]
    title = pretty(disease)
    if intent == "symptoms":
        body = f"Common signs of {title}:\n{_bullets(e['symptoms'])}"
    elif intent == "causes":
        body = f"What causes {title}:\n{_bullets(e['causes'])}"
    elif intent == "treatment":
        body = f"How {title} is usually managed:\n{_bullets(e['treatment'])}"
    elif intent == "prevention":
        body = f"Tips to prevent or reduce {title}:\n{_bullets(e['prevention'])}"
    elif intent == "contagious":
        body = f"Is {title} contagious?\n{e['contagious']}"
    elif intent == "see_doctor":
        body = f"When to see a doctor about {title}:\n{e['see_doctor']}"
    else:
        body = (
            f"{title}\n{e['overview']}\n\n"
            f"Common signs:\n{_bullets(e['symptoms'][:3])}\n\n"
            f"When to see a doctor:\n{e['see_doctor']}"
        )
    return f"{body}\n\n{DISCLAIMER}"


def _suggestions(disease):
    if not disease:
        return ["What diseases can you detect?", "How does the model work?", "Tell me about eczema"]
    t = pretty(disease)
    return [f"Symptoms of {t}", f"How is {t} treated?", f"Is {t} contagious?", f"When should I see a doctor?"]


def get_reply(message, disease=None, class_names=None):
    """
    message:     the user's text
    disease:     (optional) latest predicted class name from the website
    class_names: (optional) list of the model's classes, for 'list diseases'
    Returns: {"reply": str, "disease": str|None, "suggestions": [str]}
    """
    text = _norm(message or "")
    context = kb_name_for(disease)

    if not text:
        return {"reply": "Please type a question, for example: 'What is eczema?'",
                "disease": context, "suggestions": _suggestions(context)}

    if any(_norm(w) in text for w in URGENT_WORDS):
        return {"reply": URGENT_REPLY, "disease": context, "suggestions": _suggestions(context)}

    found = _find_disease(text)
    intent = _find_intent(text)
    target = found or context

    # Greetings and small talk
    if re.fullmatch(r"(hi|hello|hey|hii+|good (morning|afternoon|evening)|namaste)( there)?", text):
        hint = f" I see your latest result is {pretty(context)}; ask me anything about it." if context else ""
        return {"reply": "Hello! I'm DermaBot, the skin health assistant for this project. "
                         "Ask me about symptoms, causes, treatment or prevention of a skin condition." + hint,
                "disease": context, "suggestions": _suggestions(context)}
    if re.search(r"\b(thank|thanks|thank you|thx)\b", text):
        return {"reply": "You're welcome! Take care of your skin, and please see a dermatologist for any concerns.",
                "disease": context, "suggestions": _suggestions(context)}
    if not found and re.search(r"\b(bye|goodbye|see you)\b", text):
        return {"reply": "Goodbye! Stay healthy.", "disease": context, "suggestions": []}

    # Questions about the project / model
    if not found and re.search(r"\b(which|what|list|all)\b.*\b(disease|diseases|classes|conditions|detect)\b", text):
        names = class_names or list(KB.keys())
        return {"reply": f"The model can classify {len(names)} skin conditions:\n"
                         + _bullets([pretty(n) for n in names])
                         + "\n\nUpload a photo on the Predict page to try it.",
                "disease": context, "suggestions": _suggestions(context)}
    if not found and re.search(r"\b(model|accuracy|accurate|how does (it|this) work|how do you work|mobilenet|cnn|ai)\b", text):
        return {"reply": "The prediction model is MobileNetV2 (a pre-trained convolutional neural network) "
                         "fine-tuned on a Kaggle skin disease dataset with 22 classes. On unseen test images it "
                         "reached 46.44% accuracy, and the correct disease appeared in its top 3 for 71.67% of "
                         "images. That is why the website shows three possible conditions.\n\n" + DISCLAIMER,
                "disease": context, "suggestions": _suggestions(context)}
    if not found and re.search(r"\b(how (to|do i) use|upload|help)\b", text):
        return {"reply": "How to use this website:\n"
                         "• Go to the Predict page and upload a clear, close-up photo of the skin area (PNG, JPG, JPEG or WEBP, up to 8 MB)\n"
                         "• Click Analyse image to see the top 3 possible conditions\n"
                         "• Click 'Ask DermaBot' to learn more about a result here\n\n" + DISCLAIMER,
                "disease": context, "suggestions": _suggestions(context)}

    if target:
        return {"reply": _answer(target, intent or "overview"), "disease": target,
                "suggestions": _suggestions(target)}

    return {"reply": "Sorry, I didn't understand that. You can ask things like:\n"
                     "• What is psoriasis?\n• What are the symptoms of tinea?\n"
                     "• How can I prevent acne?\n• Which diseases can you detect?",
            "disease": None, "suggestions": _suggestions(None)}


if __name__ == "__main__":
    print("DermaBot (type 'quit' to exit)")
    while True:
        msg = input("You: ")
        if msg.strip().lower() in {"quit", "exit"}:
            break
        print("Bot:", get_reply(msg)["reply"], "\n")
