"""Localized health headlines and actionable hints for farmers and workshop owners."""

from __future__ import annotations

from app.schemas import FaultType, HealthStatus, LanguageCode

TIER_TO_STATUS: dict[int, HealthStatus] = {
    0: "healthy",
    1: "needs_attention",
    2: "about_to_fail",
}

_MESSAGES: dict[LanguageCode, dict[int, dict[FaultType, tuple[str, str]]]] = {
    "en": {
        0: {
            "healthy": (
                "Motor running smoothly",
                "No action needed. Continue normal operation.",
            ),
            "imbalance": (
                "Motor running smoothly",
                "Slight vibration within safe limits. Keep mounts clean.",
            ),
            "misalignment": (
                "Motor running smoothly",
                "Normal alignment. No immediate action needed.",
            ),
            "bearing_wear": (
                "Motor running smoothly",
                "Bearings within normal range. Keep lubrication schedule.",
            ),
            "demagnetization": (
                "Motor running smoothly",
                "Motor current steady. Continue normal operation.",
            ),
            "combined": (
                "Motor running smoothly",
                "Operating within safe limits.",
            ),
        },
        1: {
            "healthy": (
                "Check motor heat and load",
                "Motor load or heat is rising. Let it cool and check vents.",
            ),
            "imbalance": (
                "Rotor imbalance detected",
                "Clean fan or impeller blades and tighten mounting bolts soon.",
            ),
            "misalignment": (
                "Shaft misalignment detected",
                "Check belt tension and shaft coupling alignment today.",
            ),
            "bearing_wear": (
                "Early bearing wear",
                "Grease bearings soon and watch for grinding noise.",
            ),
            "demagnetization": (
                "Motor torque ripple high",
                "Inspect controller wires, reduce heavy load, and check heat.",
            ),
            "combined": (
                "Multiple motor stresses",
                "Reduce load and inspect shaft, bearings, and mounts soon.",
            ),
        },
        2: {
            "healthy": (
                "High motor stress alert",
                "Stop motor now. Check overload, cooling vents, and power.",
            ),
            "imbalance": (
                "Severe rotor imbalance",
                "Stop motor soon! Check broken blades, debris, and loose bolts.",
            ),
            "misalignment": (
                "Critical shaft misalignment",
                "Stop motor now! Realign shaft coupling and pulley before use.",
            ),
            "bearing_wear": (
                "Severe bearing damage",
                "Stop motor immediately! Replace worn bearing before seizure.",
            ),
            "demagnetization": (
                "Motor magnet/winding fault",
                "Stop motor now! Check overheating, phase wires, and rotor.",
            ),
            "combined": (
                "Critical multi-fault alert",
                "Shut down motor immediately to prevent costly breakdown.",
            ),
        },
    },
    "hi": {
        0: {
            "healthy": (
                "मोटर ठीक चल रही है",
                "अभी किसी काम की ज़रूरत नहीं। सामान्य रूप से चलाएं।",
            ),
            "imbalance": (
                "मोटर ठीक चल रही है",
                "हल्का कंपन सामान्य सीमा में है। बोल्ट साफ और कसे रखें।",
            ),
            "misalignment": (
                "मोटर ठीक चल रही है",
                "शाफ्ट सीध ठीक है। अभी कोई मरम्मत ज़रूरी नहीं।",
            ),
            "bearing_wear": (
                "मोटर ठीक चल रही है",
                "बियरिंग सामान्य है। समय पर ग्रीसिंग करते रहें।",
            ),
            "demagnetization": (
                "मोटर ठीक चल रही है",
                "करंट और टॉर्क सामान्य है। काम जारी रखें।",
            ),
            "combined": (
                "मोटर ठीक चल रही है",
                "मोटर सुरक्षित सीमा में चल रही है।",
            ),
        },
        1: {
            "healthy": (
                "मोटर की गर्मी और लोड जांचें",
                "तापमान या लोड बढ़ रहा है। जाली साफ करें और थोड़ा ठंडा होने दें।",
            ),
            "imbalance": (
                "रोटर असंतुलन का संकेत",
                "पंखे या ब्लेड की गंदगी साफ करें और बेस बोल्ट कसें।",
            ),
            "misalignment": (
                "शाफ्ट सीध में गड़बड़ी",
                "आज ही बेल्ट खिंचाव और कपलिंग की सीध जांच लें।",
            ),
            "bearing_wear": (
                "बियरिंग घिसने की शुरुआत",
                "जल्द बियरिंग में ग्रीस डालें और आवाज़ पर ध्यान दें।",
            ),
            "demagnetization": (
                "मोटर करंट में उतार-चढ़ाव",
                "कंट्रोलर तार जांचें, भारी लोड कम करें और गर्मी देखें।",
            ),
            "combined": (
                "मोटर पर एक से अधिक दबाव",
                "लोड कम करें और जल्द शाफ्ट, बियरिंग व बोल्ट जांचें।",
            ),
        },
        2: {
            "healthy": (
                "मोटर पर गंभीर दबाव चेतावनी",
                "मोटर तुरंत बंद करें। ओवरलोड, हवा जाली और बिजली जांचें।",
            ),
            "imbalance": (
                "गंभीर रोटर असंतुलन",
                "मोटर अभी रोकें! टूटे ब्लेड, कचरा और ढीले बोल्ट जांचें।",
            ),
            "misalignment": (
                "शाफ्ट भारी तिरछापन चेतावनी",
                "मोटर तुरंत बंद करें! चलाने से पहले शाफ्ट व पुली सीध में करें।",
            ),
            "bearing_wear": (
                "बियरिंग खराब होने का खतरा",
                "मोटर तुरंत बंद करें! जाम होने से पहले बियरिंग बदलें।",
            ),
            "demagnetization": (
                "मोटर मैग्नेट या वाइंडिंग दोष",
                "मोटर अभी रोकें! अधिक गर्मी, फेज़ तार और रोटर की जांच करें।",
            ),
            "combined": (
                "गंभीर बहु-दोष चेतावनी",
                "बड़े नुकसान से बचने के लिए मोटर तुरंत बंद करें।",
            ),
        },
    },
}


def get_localized_copy(
    language: str, tier: int, fault: FaultType
) -> tuple[LanguageCode, str, str]:
    """Return normalized language code, headline, and action hint."""
    lang: LanguageCode = "hi" if language == "hi" else "en"
    clamped_tier = max(0, min(2, tier))
    tier_map = _MESSAGES[lang][clamped_tier]
    headline, hint = tier_map.get(fault, tier_map["healthy"])
    return lang, headline, hint
