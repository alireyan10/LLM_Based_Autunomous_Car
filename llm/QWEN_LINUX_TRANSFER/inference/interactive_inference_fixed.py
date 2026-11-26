import os
import json
import math

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel
    HAS_LLM_DEPS = True
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    StoppingCriteria = object
    StoppingCriteriaList = None
    PeftModel = None
    HAS_LLM_DEPS = False


# System prompt matching Experiment 1 fine-tuning distribution
DEFAULT_SYSTEM_PROMPT = (
    "You are a robot command mapping assistant. Convert the given Bangla "
    "voice command into a structured JSON action plan."
)

# Standard ChatML chat template compatible with Qwen2.5 fine-tuning
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


# Allowed canonical action intents
ALLOWED_ACTIONS = {
    "MOVE_BACKWARD",
    "MOVE_FORWARD",
    "MOVE_TO_TARGET",
    "MOVE_NEAR_TARGET",
    "STOP",
    "TURN_LEFT",
    "TURN_RIGHT",
}

# Action normalization mapping (aliases -> canonical intent)
ACTION_ALIASES = {
    "MOVE_TOWARD_TARGET": "MOVE_TO_TARGET",
    "ROTATE_LEFT": "TURN_LEFT",
    "ROTATE_RIGHT": "TURN_RIGHT",
}

# Mapping of target names to Bangla and English keywords
TARGET_KEYWORD_MAP = [
    # Table (longer phrases first)
    ("dining_table", ["ডাইনিং টেবিল", "ডাইনিং টেবিলে", "ডাইনিং টেবিলের", "খাবার টেবিল", "খাবার টেবিলে", "dining table", "dining_table"]),
    ("table", ["টেবিল", "টেবিলে", "টেবিলের", "table"]),

    # Bedroom / Bed
    ("bedroom", ["বেডরুম", "বেডরুমে", "বেডরুমের", "শোবার ঘর", "শোবার ঘরে", "শোবার ঘরের", "শোবারঘর", "শোবারঘরে", "bedroom", "bed room"]),
    ("bed", ["খাট", "খাটের", "বিছানা", "বিছানায়", "বিছানার", "বেড", "বেডে", "বেডের", "bed"]),

    # Refrigerator / Kitchen
    ("refrigerator", ["রেফ্রিজারেটর", "রেফ্রিজারেটরে", "রেফ্রিজারেটরের", "ফ্রিজ", "ফ্রিজে", "ফ্রিজের", "refrigerator", "fridge"]),
    ("kitchen", ["রান্নাঘর", "রান্নাঘরে", "রান্নাঘরের", "kitchen"]),

    # Sofa / TV
    ("sofa", ["সোফা", "সোফায়", "সোফার", "sofa", "couch"]),
    ("tv", ["টিভি", "টিভির", "টেলিভিশন", "tv", "television"]),

    # Common unsupported locations (for explicit detection and rejection)
    ("bathroom", ["বাথরুম", "বাথরুমে", "বাথরুমের", "গোসলখানা", "bathroom", "washroom"]),
    ("garage", ["গ্যারেজ", "গ্যারেজে", "garage"]),
    ("balcony", ["বারান্দা", "বারান্দায়", "balcony"]),
    ("garden", ["বাগান", "বাগানে", "garden"]),
    ("storage_room", ["স্টোরেজ রুম", "স্টোরেজ", "storage room", "storage"]),
]


def extract_targets_from_command(command: str):
    """
    Extract target location names from a command string in sequential order of occurrence.
    """
    if not command or not isinstance(command, str):
        return []
    cmd_lower = command.lower()
    matches = []
    for target, keywords in TARGET_KEYWORD_MAP:
        for kw in keywords:
            kw_lower = kw.lower()
            start = 0
            while True:
                idx = cmd_lower.find(kw_lower, start)
                if idx == -1:
                    break
                matches.append((idx, idx + len(kw_lower), target))
                start = idx + len(kw_lower)
    # Sort matches by start position ascending, then longer span first
    matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
    non_overlapping = []
    last_end = -1
    for start, end, target in matches:
        if start >= last_end:
            non_overlapping.append((start, end, target))
            last_end = end
    return [m[2] for m in non_overlapping]


BANGLA_DIGITS_MAP = {
    '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
    '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
}


def parse_bangla_or_english_number(num_str: str):
    """Convert Bangla or English number string (including decimals like ১.৫ or 1.5) to float."""
    if not num_str or not isinstance(num_str, str):
        return None
    res = []
    for ch in num_str.strip():
        if ch in BANGLA_DIGITS_MAP:
            res.append(BANGLA_DIGITS_MAP[ch])
        elif ch in '.0123456789':
            res.append(ch)
    try:
        val = float(''.join(res))
        return val if math.isfinite(val) else None
    except ValueError:
        return None


def extract_distances_from_command(command: str):
    """
    Extract explicit motion distances (in meters) from a Bangla or English command string.
    Preserves explicit distances such as 1.5m, 2m, ১.৫ মিটার, ২ মিটার, ০.৫ মিটার, etc.
    """
    if not command or not isinstance(command, str):
        return []
    import re
    dist_pattern = re.compile(
        r'([০-৯0-9]+(?:\.[০-৯0-9]+)?)\s*(?:মিটার|মিটারের|মি\b|মি\.|m\b|meter|meters)',
        re.IGNORECASE
    )
    results = []
    for m in dist_pattern.finditer(command):
        val = parse_bangla_or_english_number(m.group(1))
        if val is not None and val > 0:
            results.append(val)
    return results


def extract_angles_from_command(command: str):
    """
    Extract explicit turn angles (in degrees) from a Bangla or English command string.
    Preserves explicit angles such as ৯০ ডিগ্রি, 90 deg, ১৮০ ডিগ্রি, 180°, ৪৫ ডিগ্রি, etc.
    """
    if not command or not isinstance(command, str):
        return []
    import re
    angle_pattern = re.compile(
        r'([০-৯0-9]+(?:\.[০-৯0-9]+)?)\s*(?:ডিগ্রি|ডিগ্রী|degree|degrees|deg\b|°)',
        re.IGNORECASE
    )
    results = []
    for m in angle_pattern.finditer(command):
        val = parse_bangla_or_english_number(m.group(1))
        if val is not None and val > 0:
            results.append(val)
    return results


def configure_chat_template(tokenizer):
    """
    Ensure the tokenizer has the compatible Qwen ChatML template configured.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None) is None:
        try:
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE
        except Exception:
            pass
    return tokenizer


def format_prompt(cmd, system_prompt=DEFAULT_SYSTEM_PROMPT, tokenizer=None):
    """
    Construct the prompt matching the exact fine-tuning ChatML format:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {cmd}<|im_end|>
    <|im_start|>assistant

    Handles:
    - Tokenizer with valid chat_template.
    - Tokenizer where chat_template is None (configures or passes QWEN_CHAT_TEMPLATE).
    - Tokenizer is None or apply_chat_template unavailable (direct plain-text ChatML construction).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cmd}
    ]

    # If tokenizer has a valid chat_template set, try apply_chat_template
    if tokenizer is not None and getattr(tokenizer, "chat_template", None) is not None:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass

    # If tokenizer has apply_chat_template but chat_template is None, pass explicit chat_template
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                chat_template=QWEN_CHAT_TEMPLATE,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass

    # Direct fallback / plain-text construction matching the fine-tuning format
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{cmd}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_llm_output(raw_text):
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1]
    if "```" in text:
        text = text.split("```")[0]

    text = text.strip()

    start_idx = text.find('{')
    if start_idx == -1:
        return None, False

    # First attempt clean parse between first { and last }
    end_idx = text.rfind('}')
    if end_idx != -1 and end_idx >= start_idx:
        candidate = text[start_idx:end_idx+1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data, True
        except json.JSONDecodeError:
            pass

    # If full parse failed (e.g. repeated trailing JSON fragments),
    # extract the first fully-formed JSON object using raw_decode
    try:
        decoder = json.JSONDecoder()
        data, _ = decoder.raw_decode(text[start_idx:])
        if isinstance(data, dict):
            return data, True
    except (json.JSONDecodeError, ValueError):
        pass

    return None, False

class JsonStoppingCriteria(StoppingCriteria):
    """
    Custom stopping criteria that halts generation as soon as a valid JSON object is fully formed.
    This safely prevents the model from generating garbage text or repeated fragments after the closing brace.
    """
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores=None, **kwargs):
        generated_ids = input_ids[0][self.prompt_length:]
        if len(generated_ids) == 0:
            if torch is not None and isinstance(input_ids, torch.Tensor):
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            return False

        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        is_done = False
        if '}' in text:
            start_idx = text.find('{')
            if start_idx != -1:
                sub = text[start_idx:].strip()
                try:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(sub)
                    if isinstance(obj, dict):
                        is_done = True
                except (json.JSONDecodeError, ValueError):
                    pass

        if torch is not None and isinstance(input_ids, torch.Tensor):
            return torch.full((input_ids.shape[0],), is_done, dtype=torch.bool, device=input_ids.device)
        return is_done

def normalize_and_validate_plan(parsed_json, locations_path=None, command=None):
    """
    Validate and normalize action plan into the consistent, strictly validated JSON schema.

    Requirements:
    1. Every sequential command must use:
       {"intent": "SEQUENTIAL", "step_count": <int>, "action_sequence": [...]}
    2. Every action must be a structured JSON object, never a bare string.
    3. Exact action names: MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT, MOVE_TO_TARGET, MOVE_NEAR_TARGET, STOP.
    4. Defaults applied only to omitted parameters:
       - MOVE_FORWARD distance: 1.0 m
       - MOVE_BACKWARD distance: 1.0 m
       - TURN_LEFT angle: 90.0 deg
       - TURN_RIGHT angle: 90.0 deg
    5. Maximum limits:
       - Movement distance: max 2.0 m
       - Absolute turn angle: max 180.0 deg
    6. STOP requires no distance or angle.
    7. MOVE_TO_TARGET / MOVE_NEAR_TARGET requires a known target from locations.yaml or legacy aliases.
       Never invent unknown target names.
    8. Explicit invalid values are rejected.
    9. step_count is recalculated from the final action_sequence.
    10. Multi-target sequential commands resolve distinct targets sequentially from command context,
        avoiding duplicate target assignment across distinct target actions.
    """
    import math

    if not isinstance(parsed_json, dict):
        return False, None, f"Expected JSON object (dict), got {type(parsed_json).__name__}."

    if command is None and isinstance(parsed_json, dict):
        command = parsed_json.get("command") or parsed_json.get("text")

    extracted_targets = extract_targets_from_command(command) if command else []
    extracted_distances = extract_distances_from_command(command) if command else []
    extracted_angles = extract_angles_from_command(command) if command else []

    raw_intent = str(parsed_json.get("intent", "")).strip().upper()

    # Extract top-level target if provided
    top_target = ""
    if "target" in parsed_json and parsed_json["target"] is not None and not isinstance(parsed_json["target"], bool):
        top_target = str(parsed_json["target"]).strip().lower()

    # Normalize plan-level intent if it exists in ACTION_ALIASES
    intent = ACTION_ALIASES.get(raw_intent, raw_intent)

    target_intents = {"MOVE_TO_TARGET", "MOVE_NEAR_TARGET", "MOVE_TOWARD_TARGET"}

    # Single-action normalization (preserves single-target navigation and single motions)
    if "action_sequence" not in parsed_json:
        if intent in ALLOWED_ACTIONS:
            single_action_dict = dict(parsed_json)
            single_action_dict["intent"] = intent
            if intent in target_intents and not single_action_dict.get("target"):
                chosen_target = extracted_targets[0] if extracted_targets else top_target
                if chosen_target:
                    single_action_dict["target"] = chosen_target
            raw_actions = [single_action_dict]
            intent = "SEQUENTIAL"
        else:
            return False, None, f"Unsupported plan intent '{raw_intent}'. Expected 'SEQUENTIAL' or supported action intent."
    else:
        raw_actions = parsed_json.get("action_sequence")
        if not isinstance(raw_actions, list):
            return False, None, f"Field 'action_sequence' must be a list, got {type(raw_actions).__name__}."

        if len(raw_actions) == 0:
            return False, None, "Field 'action_sequence' must not be empty."

        if intent in ALLOWED_ACTIONS:
            # Single-action plan that also included action_sequence (such as model output for MOVE_NEAR_TARGET / MOVE_TO_TARGET)
            intent = "SEQUENTIAL"
        elif intent != "SEQUENTIAL":
            return False, None, f"Expected top-level intent 'SEQUENTIAL', got '{raw_intent}'."

    # Load known locations
    known_locations = {"refrigerator", "bed", "sofa", "table"}
    legacy_aliases = {"kitchen": "refrigerator", "bedroom": "bed", "tv": "sofa", "dining_table": "table"}

    if locations_path is None:
        # Check standard locations
        curr_file = os.path.abspath(__file__)
        candidate = os.path.normpath(os.path.join(curr_file, "..", "..", "..", "src", "location_navigation", "config", "locations.yaml"))
        if os.path.isfile(candidate):
            locations_path = candidate
        else:
            default_loc = os.path.expanduser("~/case_study/src/location_navigation/config/locations.yaml")
            if os.path.isfile(default_loc):
                locations_path = default_loc

    if locations_path and os.path.isfile(locations_path):
        try:
            with open(locations_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#") or not stripped:
                        continue
                    if line.startswith("  ") and not line.startswith("    ") and stripped.endswith(":"):
                        loc_name = stripped[:-1].strip().lower()
                        if loc_name:
                            known_locations.add(loc_name)
        except Exception:
            pass

    # Resolve target assignments across sequential actions
    needs_target_indices = []
    for idx, act in enumerate(raw_actions):
        if isinstance(act, str):
            act_str = act.strip()
            parts = act_str.split(maxsplit=1)
            first_upper = parts[0].upper() if parts else ""
            arg_str = parts[1].strip() if len(parts) > 1 else ""
            canonical_upper = ACTION_ALIASES.get(first_upper, first_upper)
            if (first_upper in target_intents or canonical_upper in target_intents or act_str.upper() in target_intents) and not arg_str:
                needs_target_indices.append(idx)
        elif isinstance(act, dict):
            act_intent = str(act.get("intent", "")).strip().upper()
            canonical_intent = ACTION_ALIASES.get(act_intent, act_intent)
            if (act_intent in target_intents or canonical_intent in target_intents) and not act.get("target"):
                needs_target_indices.append(idx)

    num_needed = len(needs_target_indices)
    assigned_targets = {}
    if num_needed == 1:
        chosen = extracted_targets[0] if extracted_targets else top_target
        if chosen:
            assigned_targets[needs_target_indices[0]] = chosen
    elif num_needed > 1:
        if len(extracted_targets) < num_needed:
            return False, None, (
                f"Action sequence contains {num_needed} target actions, but only {len(extracted_targets)} "
                f"target(s) could be resolved from command. Cannot determine distinct targets for all steps."
            )
        for i, idx in enumerate(needs_target_indices):
            assigned_targets[idx] = extracted_targets[i]

    normalized_actions = []
    dist_idx = 0
    angle_idx = 0

    for idx, act in enumerate(raw_actions):
        step_label = f"Step {idx + 1}"

        # If action is a bare string from the LLM, normalize it into a candidate structured object
        if isinstance(act, str):
            act_str = act.strip()
            parts = act_str.split(maxsplit=1)
            first_upper = parts[0].upper() if parts else ""
            arg_str = parts[1].strip() if len(parts) > 1 else ""
            target_cand = arg_str.lower() if arg_str else assigned_targets.get(idx, "")

            if first_upper in ACTION_ALIASES:
                canonical = ACTION_ALIASES[first_upper]
                act = {"intent": canonical}
                if target_cand:
                    act["target"] = target_cand
                elif canonical in {"TURN_LEFT", "TURN_RIGHT"} and arg_str:
                    try:
                        act["angle"] = float(arg_str)
                    except ValueError:
                        pass
                elif canonical in {"MOVE_FORWARD", "MOVE_BACKWARD"} and arg_str:
                    try:
                        act["distance"] = float(arg_str)
                    except ValueError:
                        pass
            elif act_str.upper() in ACTION_ALIASES:
                canonical = ACTION_ALIASES[act_str.upper()]
                act = {"intent": canonical}
                if target_cand:
                    act["target"] = target_cand
            elif first_upper in ALLOWED_ACTIONS:
                act = {"intent": first_upper}
                if first_upper in target_intents:
                    if target_cand:
                        act["target"] = target_cand
                elif first_upper in {"MOVE_FORWARD", "MOVE_BACKWARD"} and arg_str:
                    try:
                        act["distance"] = float(arg_str)
                    except ValueError:
                        pass
                elif first_upper in {"TURN_LEFT", "TURN_RIGHT"} and arg_str:
                    try:
                        act["angle"] = float(arg_str)
                    except ValueError:
                        pass
            elif act_str.upper() in ALLOWED_ACTIONS:
                act = {"intent": act_str.upper()}
                if act_str.upper() in target_intents and target_cand:
                    act["target"] = target_cand
            elif act_str.lower() in known_locations or act_str.lower() in legacy_aliases:
                act = {"intent": "MOVE_TO_TARGET", "target": act_str.lower()}
            else:
                act = {"intent": act_str.upper()}

        if not isinstance(act, dict):
            return False, None, f"{step_label}: Action must be a structured JSON object (dict) or supported action string, got {type(act).__name__}."

        # Action normalization before action validation:
        # 1. Read the action intent.
        raw_act_intent = str(act.get("intent", "")).strip().upper()
        if not raw_act_intent:
            return False, None, f"{step_label}: Action missing required 'intent' key."

        # 2. If it exists in ACTION_ALIASES, replace it with the corresponding canonical intent.
        # 4. Preserve the target associated with each action.
        act = dict(act)
        normalized_intent = ACTION_ALIASES.get(raw_act_intent, raw_act_intent)
        act["intent"] = normalized_intent
        if normalized_intent in target_intents and ("target" not in act or not act["target"]):
            target_cand = assigned_targets.get(idx, "")
            if target_cand:
                act["target"] = target_cand

        # 3. Validate the normalized intent against ALLOWED_ACTIONS.
        if normalized_intent not in ALLOWED_ACTIONS:
            return False, None, f"{step_label}: Unsupported action intent '{raw_act_intent}'. Allowed: {sorted(ALLOWED_ACTIONS)}."

        act_intent = normalized_intent

        # MOVE_FORWARD
        if act_intent == "MOVE_FORWARD":
            cmd_dist = extracted_distances[dist_idx] if dist_idx < len(extracted_distances) else None
            dist_idx += 1
            if "distance" not in act or act["distance"] is None:
                dist = cmd_dist if cmd_dist is not None else 1.0  # default 1.0 m
            else:
                dist = act["distance"]
                if cmd_dist is not None and (dist == 1.0 and cmd_dist != 1.0):
                    dist = cmd_dist
                if isinstance(dist, bool) or not isinstance(dist, (int, float)):
                    return False, None, f"{step_label}: MOVE_FORWARD 'distance' must be numeric, got {type(dist).__name__}."
                if dist <= 0 or not math.isfinite(dist):
                    return False, None, f"{step_label}: MOVE_FORWARD 'distance' must be positive, got {dist}."
                if dist > 2.0:
                    return False, None, f"{step_label}: MOVE_FORWARD 'distance' {dist}m exceeds maximum safety limit 2.0m."
            normalized_actions.append({"intent": "MOVE_FORWARD", "distance": float(dist)})

        # MOVE_BACKWARD
        elif act_intent == "MOVE_BACKWARD":
            cmd_dist = extracted_distances[dist_idx] if dist_idx < len(extracted_distances) else None
            dist_idx += 1
            if "distance" not in act or act["distance"] is None:
                dist = cmd_dist if cmd_dist is not None else 1.0  # default 1.0 m
            else:
                dist = act["distance"]
                if cmd_dist is not None and (dist == 1.0 and cmd_dist != 1.0):
                    dist = cmd_dist
                if isinstance(dist, bool) or not isinstance(dist, (int, float)):
                    return False, None, f"{step_label}: MOVE_BACKWARD 'distance' must be numeric, got {type(dist).__name__}."
                if dist <= 0 or not math.isfinite(dist):
                    return False, None, f"{step_label}: MOVE_BACKWARD 'distance' must be positive, got {dist}."
                if dist > 2.0:
                    return False, None, f"{step_label}: MOVE_BACKWARD 'distance' {dist}m exceeds maximum safety limit 2.0m."
            normalized_actions.append({"intent": "MOVE_BACKWARD", "distance": float(dist)})

        # TURN_LEFT / TURN_RIGHT
        elif act_intent in {"TURN_LEFT", "TURN_RIGHT"}:
            cmd_ang = extracted_angles[angle_idx] if angle_idx < len(extracted_angles) else None
            angle_idx += 1
            if "angle" not in act or act["angle"] is None:
                ang = cmd_ang if cmd_ang is not None else 90.0  # default 90.0 deg
                unit = "deg"
            else:
                ang = act["angle"]
                if cmd_ang is not None and (ang == 90.0 and cmd_ang != 90.0):
                    ang = cmd_ang
                if isinstance(ang, bool) or not isinstance(ang, (int, float)):
                    return False, None, f"{step_label}: {act_intent} 'angle' must be numeric, got {type(ang).__name__}."
                if ang <= 0 or not math.isfinite(ang):
                    return False, None, f"{step_label}: {act_intent} 'angle' must be positive, got {ang}."
                unit = str(act.get("unit", "deg")).strip().lower()
                if unit in {"rad", "radians", "radian"}:
                    if ang > (math.pi + 1e-4):
                        return False, None, f"{step_label}: {act_intent} 'angle' {ang} rad exceeds maximum limit of pi rad (180 deg)."
                else:
                    unit = "deg"
                    if ang > 180.0:
                        return False, None, f"{step_label}: {act_intent} 'angle' {ang} deg exceeds maximum safety limit 180.0 deg."
            norm_turn = {"intent": act_intent, "angle": float(ang)}
            if unit != "deg":
                norm_turn["unit"] = unit
            normalized_actions.append(norm_turn)

        # MOVE_TO_TARGET / MOVE_NEAR_TARGET / MOVE_TOWARD_TARGET
        elif act_intent in target_intents:
            if "target" not in act or act["target"] is None or isinstance(act["target"], bool):
                return False, None, f"{step_label}: {act_intent} requires a 'target' parameter."
            target = str(act["target"]).strip().lower()
            if not target:
                return False, None, f"{step_label}: {act_intent} 'target' must be a non-empty string."
            # Check target in known_locations or legacy_aliases
            canonical = legacy_aliases.get(target, target)
            if (target not in known_locations) and (canonical not in known_locations):
                return False, None, f"{step_label}: Target '{target}' is unknown. Never invent unknown target names."
            norm_target_act = {"intent": act_intent, "target": target}
            if act_intent == "MOVE_NEAR_TARGET":
                standoff = act.get("standoff", act.get("standoff_distance"))
                if standoff is not None:
                    try:
                        norm_target_act["standoff"] = float(standoff)
                    except (ValueError, TypeError):
                        pass
            normalized_actions.append(norm_target_act)

        # STOP (Requirement 6: STOP requires no distance or angle)
        elif act_intent == "STOP":
            normalized_actions.append({"intent": "STOP"})

    # Requirement 9: Recalculate step_count from final action_sequence
    normalized_plan = {
        "intent": "SEQUENTIAL",
        "step_count": len(normalized_actions),
        "action_sequence": normalized_actions,
    }
    if len(normalized_actions) == 1 and normalized_actions[0].get("intent") in target_intents:
        normalized_plan["target"] = normalized_actions[0]["target"]

    return True, normalized_plan, ""



def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adapter_path = os.path.join(base_dir, "adapter")
    model_name = "Qwen/Qwen2.5-3B"

    print("="*50)
    print("Initializing Experiment 1 Interactive Inference (FIXED)...")
    print("="*50)

    # Load Tokenizer
    tokenizer_source = adapter_path if os.path.isdir(adapter_path) and os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json")) else model_name
    print(f"Loading tokenizer from {tokenizer_source}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    configure_chat_template(tokenizer)

    # Resolve EOS tokens: include both <|endoftext|> and ChatML <|im_end|>
    eos_token_ids = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_token_ids.extend(tokenizer.eos_token_id)
        else:
            eos_token_ids.append(tokenizer.eos_token_id)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in eos_token_ids:
        eos_token_ids.append(im_end_id)

    # Load Model
    print(f"Loading base model in 4-bit...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )

    # Load Adapter
    print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("\nModel successfully loaded! Ready for inference.")
    print("Type 'exit' to quit.\n")

    # Original concise system prompt matching Experiment 1 fine-tuning distribution
    system_prompt = DEFAULT_SYSTEM_PROMPT

    while True:
        try:
            cmd = input("\nEnter Bangla command: ").strip()
            if cmd.lower() == 'exit':
                break
            if not cmd:
                continue

            text = format_prompt(cmd, system_prompt=system_prompt, tokenizer=tokenizer)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            prompt_length = inputs.input_ids.shape[1]

            # Create custom stopping criteria for this specific prompt
            stopping_criteria = StoppingCriteriaList([JsonStoppingCriteria(tokenizer, prompt_length)])

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=eos_token_ids if eos_token_ids else tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria # Stop dynamically when JSON is complete
                )

            generated_tokens = outputs[0][prompt_length:]
            prediction_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            raw_parsed_json, json_decoded = parse_llm_output(prediction_str)

            print("\n--- INFERENCE RESULTS ---")
            print(f"1. Input: {cmd}")
            print(f"2. Raw model output:\n{prediction_str}")
            print(f"3. Parsed JSON output:")

            if not json_decoded:
                print("Error: Could not decode JSON from model output.")
                print("4. JSON validity status: Invalid (Malformed JSON)")
                print("-" * 25)
                continue

            # Strict schema validation and normalization
            is_valid, normalized_plan, val_err = normalize_and_validate_plan(raw_parsed_json, command=cmd)

            if is_valid:
                json_str = json.dumps(normalized_plan, ensure_ascii=False)
                print(json.dumps(normalized_plan, indent=4, ensure_ascii=False))
                print("4. JSON validity status: Valid (Normalized Schema)")

                # BRIDGE: Send valid JSON to the ROS 2 location_navigation node
                print("\n[Bridge] Sending intent to ROS 2 Nav2...")
                import subprocess
                nav_cmd = ["ros2", "run", "location_navigation", "llm_to_nav", json_str]

                # Check if simulation time is desired or active
                if os.environ.get("USE_SIM_TIME", "").strip().lower() in ("1", "true", "yes"):
                    nav_cmd.extend(["--ros-args", "-p", "use_sim_time:=true"])

                try:
                    subprocess.run(nav_cmd, check=True)
                except Exception as e:
                    print(f"[Bridge] Error calling ros2 command: {e}")
            else:
                print(f"Rejected: {val_err}")
                print(f"4. JSON validity status: Invalid ({val_err})")
            print("-" * 25)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
