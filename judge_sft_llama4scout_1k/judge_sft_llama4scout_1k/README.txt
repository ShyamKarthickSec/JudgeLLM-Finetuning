Judge SFT Dataset (1k) for Embodied Safety Judge

- Total: 1000
- Split: train 800, dev 100, test 100
- Label balance (overall): {'ALLOW': 400, 'DENY': 400, 'ASK_CLARIFY': 200}

Schema per record:
- id, source, instruction, expected_verdict, expected_rules, messages[]

Messages are chat-format:
system: judge policy + JSON schema requirement
user: instruction + 'Output ONLY JSON'
assistant: gold JSON output

Rule IDs:
R1_HUMAN_HARM, R2_ENV_HARM, R3_SELF_HARM, R4_SAFETY_BYPASS,
R5_UNAUTHORIZED_ACCESS, R6_PRIVACY_DATA, R7_OBFUSCATED_INTENT, R8_AMBIGUOUS_DUAL_USE

Sources used (from your uploads):
- safe_detailed_1009.jsonl, unsafe_detailed_1009.jsonl, abstract_1009.jsonl, long_horizon_1009.jsonl
- harmless data-1key.json, untargeted data.json, harmful data-1key.json (converted to injection-style samples)

Note: This is text-only phase 1.
