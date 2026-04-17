import re
import os
import base64
import json
import numpy as np

# ── Optional ML libs ─────────────────────────────────────────────────
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ── Hard slurs → always Hate Speech ─────────────────────────────────
HATE_WORDS = [
    'nigger', 'nigga', 'kike', 'spic', 'chink', 'faggot', 'fag',
    'retard', 'sandnigger', 'towelhead', 'wetback', 'cracker', 'coon',
    'tranny', 'dyke', 'gook', 'raghead', 'beaner', 'slope',
]

# ── Offensive-but-not-hate words (standalone triggers) ────────────────
# These alone → Offensive (not Hate Speech)
OFFENSIVE_WORDS = [
    'stupid', 'idiot', 'dumb', 'moron', 'imbecile', 'loser',
    'ugly', 'fat', 'gross', 'pathetic', 'worthless', 'useless',
    'asshole', 'bastard', 'bitch', 'shit', 'crap', 'damn', 'hell',
    'piss', 'screw you', 'shut up', 'go to hell', 'fuck you',
    'hate you', 'hate myself', 'hate this',
    'dumbass', 'jackass', 'jerk', 'douche', 'prick', 'dick',
    'crazy', 'insane', 'psycho', 'lunatic', 'freak',
]

# ── Targeted hate patterns → Hate Speech when matched ────────────────
PATTERNS = {
    'religion': [
        # Direct calls for violence/elimination against religious groups
        r'\b(muslims?|hindus?|christians?|jews?|sikhs?|buddhists?)\b.{0,40}\b(kill|destroy|eliminate|ban|deport|remove|hate|evil|dirty|filthy|terrorist)\b',
        r'\b(kill|destroy|ban|deport|eliminate)\b.{0,40}\b(muslims?|hindus?|christians?|jews?|sikhs?)\b',
        r'\b(islam|hinduism|christianity|judaism)\b.{0,40}\b(cancer|disease|plague|evil|destroy|ban)\b',
        r'\b(mosque|temple|church|synagogue)\b.{0,40}\b(bomb|burn|destroy|attack)\b',
        r'\b(all|every|these|those)\b.{0,20}\b(muslims?|hindus?|christians?|jews?)\b.{0,30}\b(should|must|need to)\b.{0,20}\b(die|leave|go|be removed|be killed|be banned)\b',
        r'\b(kafir|infidel)\b.{0,30}\b(kill|die|destroy|hate)\b',
    ],
    'age': [
        r'\b(old|elderly|senior|boomer|aged)\b.{0,30}\b(useless|waste|die|burden|stupid|dumb|worthless|trash)\b',
        r'\b(kids?|children|youth|millennial|gen.?z|zoomer)\b.{0,30}\b(stupid|dumb|lazy|worthless|idiot|trash|ruining)\b',
        r'\b(too old|too young)\b.{0,20}\b(work|vote|drive|live|exist|contribute)\b',
        r'\b(boomers?)\b.{0,20}\b(ruined|destroy|selfish|greedy|die|useless)\b',
    ],
    'gender': [
        r'\b(women?|female|females|girl|girls)\b.{0,40}\b(kitchen|stupid|weak|inferior|belong|shut up|property|worthless|useless|trash)\b',
        r'\b(men?|males?|boys?)\b.{0,40}\b(trash|toxic|evil|useless|worthless|pigs?|dogs?)\b',
        r'\b(feminism|feminist|feminists?)\b.{0,30}\b(cancer|disease|hate|destroy|evil|stupid|trash|ruining)\b',
        r'\b(transgender|trans|non.?binary|lgbt)\b.{0,30}\b(mental|sick|freak|fake|wrong|evil|disease|disorder|abomination)\b',
        r'\b(women?|female)\b.{0,20}\b(should not|shouldnt|cant|cannot)\b.{0,20}\b(vote|work|lead|drive|own)\b',
    ],
    'ethnicity': [
        r'\b(black|white|asian|hispanic|latino|latina|indian|arab|african|chinese|pakistani|mexican)\b.{0,40}\b(criminal|criminals|lazy|stupid|inferior|dirty|filthy|animal|animals|ape|apes|terrorist|terrorists)\b',
        r'\b(go back|leave|get out|deport)\b.{0,30}\b(country|home|where you came|your country|africa|mexico|india|pakistan)\b',
        r'\b(race|racial|ethnic)\b.{0,20}\b(inferior|superior|pure|dirty|filthy|mixing|replacement)\b',
        r'\b(white|black|asian)\b.{0,20}\b(genocide|replacement|extinction|superior|master race)\b',
        r'\b(immigrants?|illegals?|foreigners?|migrants?)\b.{0,30}\b(criminal|criminals|vermin|plague|infestation|invasion|invading|destroy|ruining|trash|scum|filth)\b',
    ],
    'bullying': [
        # Direct threats / death wishes
        r'\b(kill (your)?self|kys|you should die|go die|end your(self| life)|go kill yourself)\b',
        r'\b(i (will|gonna|going to)|we (will|gonna))\b.{0,20}\b(kill|hurt|attack|beat|destroy)\b.{0,20}\b(you|him|her|them)\b',
        # Targeted insults at a person
        r'\b(nobody likes|everyone hates|no one wants|no one likes)\b.{0,20}\b(you|him|her|them)\b',
        r'\b(you are|you\'re|ur|he is|she is)\b.{0,20}\b(worthless|pathetic|disgusting|a waste|garbage|trash|nothing|sub.?human)\b',
        r'\b(go (away|die|to hell|fuck yourself)|drop dead|get lost)\b',
        # Personal put-downs / dehumanizing
        r'\b(you are|you\'re|ur)\b.{0,20}\b(a (waste|piece of (shit|garbage|trash|crap))|subhuman|disgusting|repulsive)\b',
        r'\bwaste of (oxygen|space|breath|life|time|skin)\b',
        # Harassment / threats
        r'\b(i (will|am going to)|gonna)\b.{0,30}\b(harass|stalk|ruin|destroy|expose|dox)\b',
    ],
}

# ── Amplifiers boost score when combined with pattern matches ─────────
AMPLIFIERS = [
    r'\b(should|must|need to)\b.{0,20}\b(all\b.{0,10})?\b(die|be killed|be removed|be eliminated|be deported|be banned)\b',
    r'\ball of (them|you|us)\b',
    r'\bevery (single )?(one|last one|person|day)\b',
    r'[!?]{2,}',
    r'\b(always|never|every|all)\b.{0,20}\b(bad|evil|wrong|terrible|criminal|inferior)\b',
    r'\b(i hate (you|them|him|her|all|everything|everyone|myself|this world))\b',
    r'\b(disgusting|horrible|terrible|awful)\b.{0,20}\b(people|person|race|group)\b',
]


class HateSpeechPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.adam_clf = None
        self.adam_vectorizer = None
        self.classifier_type = 'keyword'
        self._load_model()
        self._load_adam_model()

    def _load_model(self):
        model_path     = os.path.join('models', 'bilstm_model.h5')
        tokenizer_path = os.path.join('models', 'tokenizer.pkl')

        if not TF_AVAILABLE:
            print("[INFO] TensorFlow not installed — using keyword classifier.")
            return

        if not (os.path.exists(model_path) and os.path.exists(tokenizer_path)):
            missing = []
            if not os.path.exists(model_path):
                missing.append('models/bilstm_model.h5')
            if not os.path.exists(tokenizer_path):
                missing.append('models/tokenizer.pkl')
            print(f"[INFO] Model file(s) not found: {', '.join(missing)} — using keyword classifier.")
            return

        try:
            import pickle
            self.model = tf.keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            self.classifier_type = 'bilstm'
            print("[INFO] BiLSTM model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Model load failed: {e} — falling back to keyword classifier.")

    def _load_adam_model(self):
        """Load TF-IDF + Logistic Regression (Adam/saga) classifier."""
        if not SKLEARN_AVAILABLE:
            print("[INFO] scikit-learn not installed — Adam classifier unavailable.")
            return
        clf_path = os.path.join('models', 'adam_classifier.pkl')
        vec_path  = os.path.join('models', 'adam_vectorizer.pkl')
        if not (os.path.exists(clf_path) and os.path.exists(vec_path)):
            print("[INFO] Adam model files not found — will use as blending component.")
            return
        try:
            import pickle
            with open(clf_path, 'rb') as f:
                self.adam_clf = pickle.load(f)
            with open(vec_path, 'rb') as f:
                self.adam_vectorizer = pickle.load(f)
            if self.classifier_type == 'keyword':
                self.classifier_type = 'adam'
            print("[INFO] Adam (TF-IDF + saga) classifier loaded successfully.")
        except Exception as e:
            print(f"[WARN] Adam model load failed: {e}")

    def get_classifier_type(self):
        return self.classifier_type

    # ── Text cleaning ─────────────────────────────────────────────────
    def _clean_text(self, text):
        import re
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s\'!?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # ── Adam classifier ────────────────────────────────────────────────
    def _adam_classify(self, text):
        """TF-IDF + Logistic Regression (saga/Adam) classifier."""
        if self.adam_clf is None or self.adam_vectorizer is None:
            return None
        try:
            import numpy as np
            cleaned = self._clean_text(text)
            vec = self.adam_vectorizer.transform([cleaned])
            proba = self.adam_clf.predict_proba(vec)[0]
            # Class order from training: 0=Hate, 1=Offensive, 2=Normal
            return {
                'hate_speech': float(proba[0]),
                'offensive':   float(proba[1]),
                'normal':      float(proba[2]),
            }
        except Exception as e:
            print(f"[WARN] Adam classify failed: {e}")
            return None

    # ── Text classification ───────────────────────────────────────────
    def _keyword_classify(self, text):
        text_lower = text.lower()

        # ── Step 1: Hard slur → immediate Hate Speech ────────────────
        for word in HATE_WORDS:
            if re.search(r'\b' + re.escape(word) + r's?\b', text_lower):
                scores = self._make_scores(0.97, 0.02, 0.01)
                return {
                    'prediction': 'Hate Speech',
                    'confidence': scores['hate_speech'],
                    'scores': scores,
                    'categories': self._detect_categories(text_lower),
                    'method': 'keyword'
                }

        # ── Step 2: Count targeted-hate pattern matches & amplifiers ──
        pattern_matches = sum(
            sum(1 for p in pats if re.search(p, text_lower, re.IGNORECASE))
            for pats in PATTERNS.values()
        )
        amp_count = sum(
            1 for p in AMPLIFIERS if re.search(p, text_lower, re.IGNORECASE)
        )

        # ── Step 3: Count standalone offensive words ──────────────────
        offensive_hits = sum(
            1 for w in OFFENSIVE_WORDS
            if re.search(r'\b' + re.escape(w) + r'\b', text_lower)
        )

        # ── Step 4: Score assignment ──────────────────────────────────
        combined = pattern_matches + amp_count

        if combined == 0 and offensive_hits == 0:
            # Clearly clean
            hate_score, off_score, normal_score = 0.03, 0.05, 0.92

        elif combined == 0 and offensive_hits >= 1:
            # Offensive words present, but no targeted hate pattern
            off_strength = min(0.38 + offensive_hits * 0.14, 0.82)
            hate_score   = 0.05
            off_score    = off_strength
            normal_score = max(1.0 - hate_score - off_score, 0.05)

        elif pattern_matches == 0 and amp_count >= 1:
            # Amplifiers only (aggressive tone, not targeted)
            hate_score, off_score, normal_score = 0.10, 0.55, 0.35

        elif pattern_matches == 1 and amp_count == 0 and offensive_hits <= 1:
            # Single targeted pattern, mild — but check if the match is
            # a "soft" ethnicity/gender word combined with just 'ruining'/'worthless'
            # Those border cases → Offensive, not Hate Speech
            soft_match = self._is_soft_pattern_match(text_lower)
            if soft_match:
                hate_score, off_score, normal_score = 0.20, 0.62, 0.18
            else:
                hate_score, off_score, normal_score = 0.55, 0.35, 0.10

        elif pattern_matches == 1 and (amp_count >= 1 or offensive_hits >= 2):
            # Single pattern escalated
            hate_score, off_score, normal_score = 0.58, 0.32, 0.10

        elif pattern_matches == 2 and amp_count == 0:
            # Two targeted patterns
            hate_score, off_score, normal_score = 0.68, 0.22, 0.10

        else:
            # Many patterns / amplifiers → strong hate speech
            raw = min(0.72 + pattern_matches * 0.06 + amp_count * 0.04, 0.97)
            hate_score   = raw
            off_score    = round(max((1.0 - raw) * 0.6, 0.02), 3)
            normal_score = round(max(1.0 - hate_score - off_score, 0.01), 3)

        # ── Step 5: Derive prediction ─────────────────────────────────
        scores = self._make_scores(hate_score, off_score, normal_score)

        if scores['hate_speech'] >= 0.50:
            prediction = 'Hate Speech'
            confidence = scores['hate_speech']
        elif scores['offensive'] >= 0.35 or scores['hate_speech'] >= 0.22:
            prediction = 'Offensive'
            confidence = max(scores['offensive'], scores['hate_speech'])
        else:
            prediction = 'Normal'
            confidence = scores['normal']

        return {
            'prediction': prediction,
            'confidence': round(confidence, 3),
            'scores': scores,
            'categories': self._detect_categories(text_lower),
            'method': 'keyword'
        }

    def _is_soft_pattern_match(self, text_lower):
        """
        Returns True when the ONLY pattern match is a borderline one that
        should be Offensive rather than Hate Speech.
        E.g. "immigrants are ruining everything" or "you are worthless" (no group target).
        """
        # Hard dehumanizing words that always escalate to Hate Speech
        HARD_WORDS = ['vermin', 'scum', 'filth', 'trash', 'criminal', 'plague',
                      'infestation', 'parasite', 'animals', 'apes', 'inferior', 'terrorist']
        has_hard_word = any(re.search(r'\b' + w + r'\b', text_lower) for w in HARD_WORDS)
        if has_hard_word:
            return False

        SOFT_PATTERNS = [
            r'\b(immigrants?|foreigners?|migrants?)\b.{0,30}\b(ruining|bad|wrong|problem|annoying)\b',
            r'\byou (are|re|r).{0,15}\b(worthless|useless|pathetic|disgusting)\b',
        ]
        return any(re.search(p, text_lower, re.IGNORECASE) for p in SOFT_PATTERNS)

    def _detect_categories(self, text_lower):
        found = [cat for cat, pats in PATTERNS.items()
                 if any(re.search(p, text_lower, re.IGNORECASE) for p in pats)]
        return found if found else ['none']

    def _make_scores(self, hate, offensive, normal):
        # Clamp all to [0, 1]
        hate     = max(0.0, min(1.0, hate))
        offensive = max(0.0, min(1.0, offensive))
        normal   = max(0.0, min(1.0, normal))
        total = hate + offensive + normal
        if total == 0:
            total = 1.0  # fallback: avoid division by zero
        return {
            'hate_speech': round(hate / total, 3),
            'offensive':   round(offensive / total, 3),
            'normal':      round(normal / total, 3),
        }

    def _bilstm_classify(self, text):
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=100)
            probs = self.model.predict(padded, verbose=0)[0]
            classes = ['Hate Speech', 'Offensive', 'Normal']
            idx = int(np.argmax(probs))
            return {
                'prediction': classes[idx],
                'confidence': round(float(probs[idx]), 3),
                'scores': {
                    'hate_speech': round(float(probs[0]), 3),
                    'offensive': round(float(probs[1]), 3),
                    'normal': round(float(probs[2]), 3)
                },
                'categories': self._detect_categories(text.lower()),
                'method': 'bilstm'
            }
        except Exception as e:
            print(f"[WARN] BiLSTM failed: {e}")
            return self._keyword_classify(text)

    def _claude_classify(self, text):
        """Use Claude API for semantic understanding — catches any phrasing."""
        try:
            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
            client = anthropic.Anthropic(api_key=api_key)

            prompt = f"""You are an expert hate speech and offensive content classifier.

Analyze the following text and classify it. Consider ALL forms of hate speech including:
- Indirect/coded language ("those people", "you know who", dog whistles)
- Sarcasm/irony used to demean groups
- Self-directed negativity (self-harm ideation, extreme self-deprecation)
- Threats, bullying, harassment (even subtle)
- Discrimination based on religion, race, gender, age, sexuality, nationality
- Any text intended to demean, dehumanize, or harm

Text to analyze: "{text}"

Respond ONLY with this exact JSON (no extra text, no markdown):
{{
  "prediction": "Hate Speech" | "Offensive" | "Normal",
  "confidence": <float 0.0-1.0>,
  "scores": {{
    "hate_speech": <float 0.0-1.0>,
    "offensive": <float 0.0-1.0>,
    "normal": <float 0.0-1.0>
  }},
  "categories": ["religion"|"ethnicity"|"gender"|"age"|"bullying"|"self-harm"|"threat"|"none"],
  "reason": "<one sentence explanation in simple English>"
}}

Rules:
- scores must sum to exactly 1.0
- prediction must match the highest score category
- be accurate — do not over-classify normal text as offensive
- self-harm language or extreme self-hate → Hate Speech or Offensive
- casual mild swearing with no target → Offensive (low confidence)
- clearly positive/neutral text → Normal"""

            response = client.messages.create(
                model="claude-haiku-4-5-20251001",   # fast + cheap for text
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )

            raw = response.content[0].text.strip()
            # Strip any accidental markdown
            raw = re.sub(r'```json|```', '', raw).strip()
            result = json.loads(raw)

            # Validate & normalise scores so they always sum to 1.0
            s = result.get('scores', {})
            h = float(s.get('hate_speech', 0))
            o = float(s.get('offensive', 0))
            n = float(s.get('normal', 0))
            total = h + o + n or 1.0
            result['scores'] = {
                'hate_speech': round(h / total, 3),
                'offensive':   round(o / total, 3),
                'normal':      round(n / total, 3),
            }
            result['confidence'] = round(float(result.get('confidence', 0)), 3)
            result['method'] = 'claude_ai'
            if 'categories' not in result:
                result['categories'] = ['none']
            return result

        except Exception as e:
            print(f"[WARN] Claude API text classify failed: {e} — falling back to keyword")
            return None

    def _hybrid_predict(self, text):
        """Blend Adam + keyword classifier for best results."""
        import numpy as np

        # ── Direct threat override (always Hate Speech) ──────────────────
        direct_threat_patterns = [
            r'\b(kill (your)?self|kys|go kill yourself|end your(self| life))\b',
            r'\bi (will|gonna|am going to)\b.{0,20}\b(kill|murder|hurt|attack)\b.{0,10}\b(you|him|her|them)\b',
            r'\b(you should|u should)\b.{0,10}\b(die|be dead|kill yourself)\b',
        ]
        text_lower = text.lower()
        for pat in direct_threat_patterns:
            if re.search(pat, text_lower, re.IGNORECASE):
                scores = self._make_scores(0.90, 0.08, 0.02)
                return {
                    'prediction': 'Hate Speech',
                    'confidence': scores['hate_speech'],
                    'scores': scores,
                    'categories': ['bullying'],
                    'method': 'hybrid',
                    'detail': 'Direct threat detected'
                }

        # ── Hard slur check (always Hate Speech before blending) ────────────────
        for word in HATE_WORDS:
            if re.search(r'\b' + re.escape(word) + r's?\b', text_lower):
                cats = self._detect_categories(text_lower)
                scores = self._make_scores(0.97, 0.02, 0.01)
                return {
                    'prediction': 'Hate Speech',
                    'confidence': scores['hate_speech'],
                    'scores': scores,
                    'categories': cats,
                    'method': 'hybrid',
                    'detail': 'Hate slur detected'
                }

        # ── Hard targeted-hate phrases → force Hate Speech ─────────────────────
        hard_hate_phrases = [
            r'\b(deport|remove|eliminate|ban)\b.{0,30}\b(muslims?|jews?|christians?|hindus?|sikhs?|immigrants?|blacks?|mexicans?|latinos?|arabs?)\b',
            r'\b(all|every)\b.{0,10}\b(muslims?|jews?|blacks?|mexicans?|immigrants?)\b.{0,30}\b(should|must|need to)\b.{0,20}\b(go|leave|die|be removed|be deported|be eliminated)\b',
            r'\b(muslims?|jews?|blacks?|immigrants?)\b.{0,30}\b(are (all|just)?)?.{0,10}\b(terrorists?|criminals?|vermin|parasites?|animals?|scum|filth|plague|infestation)\b',
            r'\b(white genocide|racial purity|master race|ethnic cleansing|race war|gas the|death to (all|the))\b',
        ]
        for pat in hard_hate_phrases:
            if re.search(pat, text_lower, re.IGNORECASE):
                cats = self._detect_categories(text_lower)
                scores = self._make_scores(0.88, 0.10, 0.02)
                return {
                    'prediction': 'Hate Speech',
                    'confidence': scores['hate_speech'],
                    'scores': scores,
                    'categories': cats,
                    'method': 'hybrid',
                    'detail': 'Targeted hate speech: ' + (cats[0] if cats else 'general')
                }

        # ── Positive sentiment guard — prevent misclassifying clearly normal text ──
        positive_patterns = [
            r'\b(i love|love you|thank you|great job|well done|congratulations|happy birthday|good morning|good night)\b',
            r'\b(wonderful|amazing|beautiful|fantastic|excellent|awesome|brilliant|perfect|lovely)\b',
            r'^(hi|hello|hey|good morning|good evening|thanks|thank you|please|sorry|excuse me)',
        ]
        pos_hits = sum(1 for p in positive_patterns if re.search(p, text_lower, re.IGNORECASE))

        # Get keyword result
        kw = self._keyword_classify(text)
        kw_scores = kw['scores']

        # Get Adam result if available
        adam = self._adam_classify(text)

        if adam is None:
            # No Adam model — just use keyword + positivity guard
            if pos_hits >= 1 and kw['prediction'] not in ('Hate Speech',):
                if kw_scores['hate_speech'] < 0.3:
                    kw['prediction'] = 'Normal'
                    kw['confidence'] = max(kw_scores['normal'], 0.65)
            kw['method'] = 'keyword'
            return kw

        # ── Dynamic blend weights based on agreement ─────────────────
        # If keyword is strongly Normal (>0.80) but Adam disagrees → trust keyword more
        # If keyword has offensive signals → trust Adam more
        kw_normal_strong  = kw_scores['normal'] >= 0.80
        kw_has_signals    = kw_scores['offensive'] >= 0.20 or kw_scores['hate_speech'] >= 0.20
        adam_top          = max(adam.values())
        adam_top_class    = max(adam, key=adam.get)

        if kw_normal_strong and adam_top_class in ('offensive', 'hate_speech') and not kw_has_signals:
            # Keyword says clean, Adam disagrees but no real signals → trust keyword 80%
            w_adam, w_kw = 0.20, 0.80
        elif kw_has_signals and adam_top >= 0.65:
            # Both agree something is wrong → trust Adam more
            w_adam, w_kw = 0.70, 0.30
        else:
            # Default balanced blend
            w_adam, w_kw = 0.55, 0.45

        blended = {
            'hate_speech': w_adam * adam['hate_speech'] + w_kw * kw_scores['hate_speech'],
            'offensive':   w_adam * adam['offensive']   + w_kw * kw_scores['offensive'],
            'normal':      w_adam * adam['normal']       + w_kw * kw_scores['normal'],
        }

        # Normalise
        total = sum(blended.values()) or 1.0
        blended = {k: v / total for k, v in blended.items()}

        # ── Positive override: strong positive words → push toward Normal ──
        if pos_hits >= 1 and blended['hate_speech'] < 0.50:
            blended['normal']      = max(blended['normal'], 0.68)
            blended['hate_speech'] = min(blended['hate_speech'], 0.12)
            blended['offensive']   = max(0.0, 1.0 - blended['normal'] - blended['hate_speech'])
        elif pos_hits >= 2 and blended['hate_speech'] < 0.60:
            blended['normal']      = max(blended['normal'], 0.75)
            blended['hate_speech'] = min(blended['hate_speech'], 0.08)
            blended['offensive']   = max(0.0, 1.0 - blended['normal'] - blended['hate_speech'])

        scores = self._make_scores(blended['hate_speech'], blended['offensive'], blended['normal'])

        # ── Final decision ────────────────────────────────────────────
        if scores['hate_speech'] >= 0.50:
            prediction = 'Hate Speech'
            confidence = scores['hate_speech']
            detail = kw.get('categories', ['none'])
        elif scores['offensive'] >= 0.36 or scores['hate_speech'] >= 0.25:
            prediction = 'Offensive'
            confidence = max(scores['offensive'], scores['hate_speech'])
            detail = 'Offensive language'
        else:
            prediction = 'Normal'
            confidence = scores['normal']
            detail = 'No harmful content'

        cats = kw.get('categories', ['none'])
        return {
            'prediction': prediction,
            'confidence': round(confidence, 3),
            'scores': scores,
            'categories': cats,
            'method': 'adam',
            'detail': detail if isinstance(detail, str) else ', '.join(cats)
        }

    def predict(self, text):
        # Priority 1: BiLSTM model (if loaded)
        if self.classifier_type == 'bilstm' and self.model:
            return self._bilstm_classify(text)

        # Priority 2: Claude AI API (if key available)
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if ANTHROPIC_AVAILABLE and api_key:
            result = self._claude_classify(text)
            if result:
                return result

        # Priority 3: Hybrid Adam + Keyword classifier
        return self._hybrid_predict(text)

    # ── OCR helper: extract text from image bytes ─────────────────────
    def _ocr_from_bytes(self, img_bytes):
        """Extract text from raw image bytes using pytesseract."""
        if not OCR_AVAILABLE:
            return ''
        try:
            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            print(f"[WARN] OCR failed: {e}")
            return ''

    def _ocr_from_b64(self, img_b64):
        """Extract text from base64 image string."""
        try:
            img_bytes = base64.b64decode(img_b64)
            return self._ocr_from_bytes(img_bytes)
        except Exception:
            return ''

    # ── Image analysis ────────────────────────────────────────────────
    def predict_image(self, img_b64, ext='jpeg'):
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')

        # ── Path 1: Anthropic Vision API (best quality) ──────────────
        if ANTHROPIC_AVAILABLE and api_key:
            try:
                client = anthropic.Anthropic(api_key=api_key)
                media_map = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                             'png': 'image/png', 'gif': 'image/gif', 'webp': 'image/webp'}
                media_type = media_map.get(ext, 'image/jpeg')
                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=600,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64",
                                                      "media_type": media_type, "data": img_b64}},
                        {"type": "text", "text": (
                            "Analyze this image for hate speech or offensive content.\n"
                            "1. Extract any visible text.\n"
                            "2. Classify: Hate Speech / Offensive / Normal\n"
                            "3. Confidence 0.0-1.0\n"
                            "4. Categories: religion, age, gender, ethnicity, bullying\n\n"
                            "Respond ONLY in this JSON:\n"
                            '{"prediction":"Normal","confidence":0.95,'
                            '"extracted_text":"text here","categories":["none"],'
                            '"reason":"explanation"}'
                        )}
                    ]}]
                )
                raw = re.sub(r'```json|```', '', response.content[0].text).strip()
                result = json.loads(raw)
                result['method'] = 'anthropic_vision'
                return result
            except Exception as e:
                print(f"[WARN] Anthropic Vision failed: {e}, falling back to OCR")

        # ── Path 2: OCR + Keyword classifier (no API key needed) ─────
        extracted_text = self._ocr_from_b64(img_b64)
        if extracted_text:
            result = self._keyword_classify(extracted_text)
            result['extracted_text'] = extracted_text
            result['method'] = 'ocr_keyword'
            result['reason'] = f'Text extracted via OCR and classified by keyword classifier.'
            return result

        # ── Path 3: No text found, image-only fallback ────────────────
        return {
            'prediction': 'Normal',
            'confidence': 0.70,
            'extracted_text': '',
            'categories': ['none'],
            'reason': (
                'No text detected in image. Visual-only analysis not available '
                '(set ANTHROPIC_API_KEY for full AI vision analysis).'
            ),
            'scores': self._make_scores(0.05, 0.10, 0.85),
            'method': 'no_text_detected'
        }

    # ── Video analysis ────────────────────────────────────────────────
    def predict_video(self, video_path):
        # Step 1: Extract frames with OpenCV
        if not CV2_AVAILABLE:
            return {
                'prediction': 'Error',
                'confidence': 0,
                'message': 'opencv-python not installed. Run: pip install opencv-python',
                'frames_analyzed': 0,
                'method': 'cv2_missing'
            }

        frames_b64 = []
        frames_bytes = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            duration = total_frames / fps
            sample_count = min(6, max(1, int(duration)))
            sample_indices = [int(i * total_frames / sample_count) for i in range(sample_count)]

            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode('.jpg', frame)
                    raw_bytes = buf.tobytes()
                    frames_bytes.append(raw_bytes)
                    frames_b64.append(base64.b64encode(raw_bytes).decode('utf-8'))
            cap.release()
        except Exception as e:
            return {
                'prediction': 'Error',
                'confidence': 0,
                'message': f'Video read error: {e}',
                'frames_analyzed': 0,
                'method': 'video_read_error'
            }

        if not frames_b64:
            return {
                'prediction': 'Error',
                'confidence': 0,
                'message': 'No frames could be extracted. Check video file format.',
                'frames_analyzed': 0,
                'method': 'no_frames'
            }

        api_key = os.environ.get('ANTHROPIC_API_KEY', '')

        # ── Path 1: Anthropic Vision API ─────────────────────────────
        if ANTHROPIC_AVAILABLE and api_key:
            try:
                client = anthropic.Anthropic(api_key=api_key)
                content = []
                for i, fb64 in enumerate(frames_b64):
                    content.append({"type": "image",
                                    "source": {"type": "base64",
                                               "media_type": "image/jpeg",
                                               "data": fb64}})
                    content.append({"type": "text", "text": f"Frame {i+1}:"})
                content.append({"type": "text", "text": (
                    "Analyze ALL frames for hate speech or offensive content. "
                    "Extract any visible text from each frame.\n\n"
                    "Respond ONLY in this JSON:\n"
                    '{"prediction":"Normal","confidence":0.92,'
                    '"extracted_text":"text found","categories":["none"],'
                    '"reason":"explanation",'
                    '"frame_details":["frame 1: desc","frame 2: desc"]}'
                )})
                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=800,
                    messages=[{"role": "user", "content": content}]
                )
                raw = re.sub(r'```json|```', '', response.content[0].text).strip()
                result = json.loads(raw)
                result['method'] = 'anthropic_vision_video'
                result['frames_analyzed'] = len(frames_b64)
                return result
            except Exception as e:
                print(f"[WARN] Anthropic Vision video failed: {e}, falling back to OCR")

        # ── Path 2: OCR each frame + keyword classify ─────────────────
        all_texts = []
        frame_details = []

        for i, fb_bytes in enumerate(frames_bytes):
            text = self._ocr_from_bytes(fb_bytes)
            if text:
                all_texts.append(text)
                frame_details.append(f"Frame {i+1}: \"{text[:80]}\"")
            else:
                frame_details.append(f"Frame {i+1}: (no text detected)")

        combined_text = ' '.join(all_texts)

        if combined_text.strip():
            result = self._keyword_classify(combined_text)
            result['extracted_text'] = combined_text[:500]
            result['frames_analyzed'] = len(frames_b64)
            result['frame_details'] = frame_details
            result['method'] = 'ocr_keyword_video'
            result['reason'] = f'Text extracted from {len(all_texts)} frames via OCR, classified by keyword classifier.'
            return result

        # ── Path 3: No text in any frame ──────────────────────────────
        return {
            'prediction': 'Normal',
            'confidence': 0.65,
            'extracted_text': '',
            'frames_analyzed': len(frames_b64),
            'frame_details': frame_details,
            'categories': ['none'],
            'scores': self._make_scores(0.05, 0.10, 0.85),
            'reason': (
                f'{len(frames_b64)} frames analyzed. No text detected in any frame. '
                'Set ANTHROPIC_API_KEY for full visual AI analysis.'
            ),
            'method': 'no_text_video'
        }
