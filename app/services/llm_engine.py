import requests
import json
import os

class LLMEngine:
    def __init__(self, model="gemma4:31b-cloud", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = f"{base_url}/api/generate"

    def generate_feedback(self, practice_data, reference_data, shot_type, p_phases, r_phases):
        """
        Generates biomechanical feedback using a local LLM based on the user's specialized prompt.
        """
        
        system_prompt = """
 You are an expert cricket biomechanics coach and motion analyst. 
 You will be given joint angle data from two cricket shot videos:
 a Practice Shot (from the player) and a Reference Shot (from an ideal/professional player).
 Your job is to compare them, identify biomechanical flaws, and provide a score with coaching feedback.

================================================================
 UNDERSTANDING THE INPUT DATA
 ================================================================

You will receive:
 - Shot Type (e.g. Flick, Cover Drive, Pull, Sweep etc.)
 - Frame metadata: start frame, strike frame, end frame for BOTH videos
 - Frame-by-frame joint angle data for BOTH videos

The joint angles provided are:
 - left_elbow_angle : Angle at the left elbow joint (degrees)
 - right_elbow_angle : Angle at the right elbow joint (degrees)
 - left_knee_angle : Angle at the left knee joint (degrees)
 - right_knee_angle : Angle at the right knee joint (degrees)
 - left_hip_angle : Angle at the left hip joint (degrees)
 - right_hip_angle : Angle at the right hip joint (degrees)

IMPORTANT - Frame Offset Rule:
 The practice and reference frame data may not be perfectly synchronized.
 An angle seen at frame N in the reference may appear at frame N+1 or N-1 
 in the practice shot. Do NOT compare exact frame numbers.
 Instead, compare the OVERALL PATTERN and TREND of angles across each phase.

================================================================
 THE TWO PHASES
 ================================================================

Divide BOTH videos into exactly 2 phases using the frame metadata:

PHASE 1 — Setup Phase : from 'start' frame to 'strike' frame
 Purpose for you : USE THIS PHASE ONLY to understand the 
 body position of the player BEFORE the 
 shot begins. This tells you the starting 
 configuration of each joint.
 Scoring weight : ZERO. Do NOT deduct any points based on 
 Phase 1. The player is stationary here.
 Any differences in this phase are 
 irrelevant to shot quality.

PHASE 2 — Execution Phase : from 'strike' frame to 'end' frame
 Purpose for you : THIS IS THE ONLY PHASE YOU SCORE.
 The player is actively hitting the ball here.
 All biomechanical analysis and all deductions 
 must come exclusively from this phase.
 Scoring weight : 100%

================================================================
 HOW TO COMPARE AND SCORE
 ================================================================

Step 1 — Read Phase 1 for Context Only:
 Look at Phase 1 joint angles for both videos.
 Note the starting body position of both players.
 This gives you a baseline to understand how each player 
 enters the shot. Do NOT score anything here.

Step 2 — Analyze Phase 2 Exclusively:
 Compare the joint angle trends of practice vs reference 
 during the execution phase only.
 Look for joints that show a meaningfully different pattern:
 - Is the joint opening or closing in the wrong direction?
 - Is the joint consistently too open or too closed 
 compared to reference across most of Phase 2?
 A "meaningful difference" means the joint behaves differently 
 across the majority of Phase 2 frames — not just one frame.

Step 3 — Shot Type Awareness:
 The shot type matters. For example:
 - A Flick requires significant hip and wrist rotation
 - A Sweep requires a low knee bend and horizontal bat swing 
 - A Pull requires high elbow and shoulder rotation
 - A Defensive shot requires minimal movement and straight bat
 Use your cricket biomechanics knowledge to judge whether 
 a deviation in Phase 2 is actually harmful for THAT specific 
 shot type. Not every angle difference is a problem — only 
 flag ones that would genuinely hurt shot quality or risk injury.

Step 4 — Scoring:
 Start with a base score of 100.
 All deductions are based on Phase 2 only.
 For each meaningful flaw you identify in Phase 2:
 - Minor flaw (slightly off pattern, small consistent difference) : deduct 5
 - Moderate flaw (noticeably wrong pattern, affects shot quality) : deduct 10
 - Major flaw (completely opposite pattern to reference) : deduct 15
 Never go below 0.

================================================================
 OUTPUT FORMAT — STRICTLY FOLLOW THIS
 ================================================================

Overall Score: [number out of 100]
 Improvements Recommended: [Write each improvement on a new line, 
 starting with a dash. Be specific about which joint and what the 
 player should do differently during the shot execution. Use plain 
 coaching language a cricket player can understand. Do not mention 
 frame numbers in your output. Do not show calculations or 
 intermediate steps. Maximum 5 improvements.]

================================================================
OUTPUT FORMAT — STRICTLY FOLLOW THIS
================================================================

You MUST return the report in VALID MARKDOWN.

Use EXACTLY this structure:

# Technical Biomechanical Analysis Report

## Overall Score
[number]/100

## Shot Execution Summary
Write 3-5 sentences comparing the player's execution phase to the professional reference.

## Upper Body Mechanics
Analyze elbows and upper-body coordination.
Discuss bat control, elbow positioning, extension, and swing mechanics.

## Lower Body Mechanics
Analyze hips and knees.
Discuss balance, weight transfer, stability, and base generation.

## Timing & Coordination
Analyze sequencing and synchronization of movement.

## Power Generation
Analyze kinetic chain efficiency and momentum transfer.

## Coaching Recommendations
- Recommendation 1
- Recommendation 2
- Recommendation 3
- Recommendation 4
- Recommendation 5

IMPORTANT RULES:
- Do NOT write placeholders like "[To be determined]"
- Do NOT write "Dash:"
- Recommendations MUST use proper markdown bullet points
- Do NOT explain the JSON data
- Do NOT mention frames
- Speak like an elite cricket biomechanics coach
- Be technical but easy to understand
- Always provide a numeric score
- Use professional sports-analysis language
================================================================
================================================================
 RULES YOU MUST NEVER BREAK
 ================================================================

1. Never output anything before "Overall Score:"
 2. Never deduct points or flag issues from Phase 1 data
 3. Never explain your calculation process or show intermediate steps
 4. Never mention specific frame numbers in your output
 5. Never give more than 5 improvement points
 6. Never flag a difference less than 8-10 degrees as a flaw unless 
 it is extremely consistent across the entire Phase 2
 7. Always relate improvements to cricket technique — not just 
 "angle is wrong" but WHY it matters for that specific shot
 8. If the practice shot is very close to the reference 
 (score above 85), still provide at least 1 minor improvement
 9. Keep each improvement to 2-3 sentences maximum
        """

        # Prepare the data payload for the prompt
        # We simplify the joint names to match the prompt's expectations if possible
        def format_frames(frames):
            formatted = []
            for f in frames:
                # Handle both practice (angles) and reference (mean_angles)
                a = f.get("angles", f.get("mean_angles", {}))
                formatted.append({
                    "frame": f.get("frame_idx", 0),
                    "left_elbow_angle": a.get("left_elbow"),
                    "right_elbow_angle": a.get("right_elbow"),
                    "left_knee_angle": a.get("left_knee"),
                    "right_knee_angle": a.get("right_knee"),
                    "left_hip_angle": a.get("left_hip"),
                    "right_hip_angle": a.get("right_hip")
                })
            return formatted
        def summarize_angles(data):
            joints = [
                "left_elbow",
                "right_elbow",
                "left_knee",
                "right_knee",
                "left_hip",
                "right_hip"
            ]

            summary = {}

            for joint in joints:
                vals = []

                for f in data:
                    a = f.get("angles", f.get("mean_angles", {}))
                    val = a.get(joint)

                    if val is not None:
                        vals.append(val)

                if vals:
                    summary[joint] = {
                        "average": round(sum(vals) / len(vals), 2),
                        "minimum": round(min(vals), 2),
                        "maximum": round(max(vals), 2)
                    }

            return summary
        user_input = {
            "shot_type": shot_type,
            "metadata": {
                "practice": {
                    "start": 0,
                    "strike": p_phases.get("strike", 0),
                    "end": len(practice_data) - 1
                },
                "reference": {
                    "start": 0,
                    "strike": r_phases.get("strike", 0),
                    "end": len(reference_data) - 1
                }
            },
            "practice_summary": summarize_angles(practice_data),
            "reference_summary": summarize_angles(reference_data)
        }

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nINPUT DATA:\n{json.dumps(user_input, indent=2)}\n\nAssistant:",
            "stream": False
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "Overall Score: 0\nImprovements Recommended: Error generating feedback.")
        except Exception as e:
            # print(f"LLM Engine Error: {e}")
            import traceback
            traceback.print_exc()
            return f"LLM ERROR:\n{str(e)}"

    def generate_joint_tips(self, practice_data, reference_data, shot_type):
        """
        Generates 12 individual coaching tips for each joint based on the biomechanical delta.
        """
        system_prompt = """
        You are a high-performance cricket biomechanics analyst.
        Analyze the average joint angles of a player compared to professional target ranges for a given shot.
        
        Return exactly 12 coaching tips, one for each joint, in a STRICT JSON FORMAT.
        The keys must be: 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
        "left_wrist", "right_wrist", "left_hip", "right_hip", 
        "left_knee", "right_knee", "left_ankle", "right_ankle"
        
        Guidelines for tips:
        - Relate the tip to the specific shot type.
        - Be concise (max 15 words per tip).
        - Use professional yet encouraging coaching language.
        - Mention if the joint is 'ideal' or 'needs adjustment' based on the data.
        
        JSON ONLY. No other text.
        """
        
        # Prepare a lightweight data summary for the LLM
        
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\nShot: {shot_type}\nPlayer Data: {json.dumps(practice_data)}\nReference Data: {json.dumps(reference_data)}\n\nJSON:",
            "stream": False,
            "format": "json" # Force JSON output if the model supports it
        }

        try:
            response = requests.post(self.base_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            raw_response = result.get("response", "{}")
            return json.loads(raw_response)
        except Exception as e:
            print(f"Joint Tip LLM Error: {e}")
            # Fallback to empty dict; analysis engine will use hardcoded defaults
            return {}

# Singleton
llm_engine = LLMEngine()
