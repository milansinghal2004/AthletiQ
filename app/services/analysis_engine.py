import os
import json
import cv2
import imageio
import numpy as np
from app.config import PROJECT_ROOT, REFERENCES_DB_PATH, OUTPUTS_DIR
# Plotting utility removed to streamline dashboard UI
from app.services.llm_engine import llm_engine

def load_references():
    if os.path.exists(REFERENCES_DB_PATH):
        with open(REFERENCES_DB_PATH, "r") as f:
            refs = json.load(f)
            for key, val in refs.items():
                # Use original path to derive stats prefix if available
                source_path = val.get("video_path_original", val["video_path"])
                base = os.path.basename(source_path)
                prefix = base.split("_reference")[0]
                val["stats_path"] = f"assets/references/{prefix}_stats.json"
                
                # Double check the stats path exists
                if not os.path.exists(os.path.join(PROJECT_ROOT, val["stats_path"])):
                    # Fallback: Try shot name as prefix
                    shot_prefix = key.lower().replace(" ", "_")
                    val["stats_path"] = f"assets/references/{shot_prefix}_stats.json"
            return refs
    return {}

def create_synced_video(practice_video, reference_video, alignment_path, progress_callback=None):
    cap_p = cv2.VideoCapture(practice_video)
    cap_r = cv2.VideoCapture(reference_video)
    fps = cap_p.get(cv2.CAP_PROP_FPS) or 30.0
    slow_fps = fps * 0.3
    
    target_h = 480
    w_p, h_p = int(cap_p.get(3)), int(cap_p.get(4))
    w_r, h_r = int(cap_r.get(3)), int(cap_r.get(4))
    scale_p, scale_r = target_h/h_p, target_h/h_r
    new_w_p, new_w_r = (int(w_p*scale_p)//2)*2, (int(w_r*scale_r)//2)*2
    
    out_path = os.path.join(OUTPUTS_DIR, "synced_comparison.mp4")
    writer = imageio.get_writer(out_path, fps=slow_fps, codec='libx264', pixelformat='yuv420p', macro_block_size=None)
    
    ref_frames = []
    while True:
        ret, f = cap_r.read()
        if not ret: break
        ref_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_r, target_h)))
    cap_r.release()
    
    p_frames = []
    while True:
        ret, f = cap_p.read()
        if not ret: break
        p_frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (new_w_p, target_h)))
    cap_p.release()
    
    total_steps = len(alignment_path)
    for step_idx, (p_idx, r_idx) in enumerate(alignment_path):
        if p_idx >= len(p_frames): p_idx = len(p_frames) - 1
        if r_idx >= len(ref_frames): r_idx = len(ref_frames) - 1
        
        frame_p = p_frames[p_idx]
        frame_r = ref_frames[r_idx]
        
        combined = np.hstack((frame_p, frame_r))
        cv2.putText(combined, "PRACTICE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(combined, "REFERENCE", (new_w_p+10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        writer.append_data(combined)
        
        if progress_callback: 
            progress_callback(0.6 + (step_idx/total_steps)*0.4, "Rendering Comparison...")
    
    writer.close()
    return out_path

def run_sync_logic(practice_data, shot_type, practice_video, extractor, sync_engine, progress_callback=None):
    if not practice_video or not shot_type or not sync_engine: return None, "Setup missing", [None]*6
    
    refs = load_references()
    if shot_type not in refs: return None, "Shot not found", [None]*6
    
    ref_info = refs[shot_type]
    ref_video = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info["video_path"]))
    stats_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("stats_path", "")))
    
    if not os.path.exists(ref_video): return None, "Ref video missing", [None]*6
    if not os.path.exists(stats_path): return None, "Ref stats missing", [None]*6
    
    with open(stats_path, "r") as f:
        stats_data = json.load(f)
        
    ref_angles_path = os.path.normpath(os.path.join(PROJECT_ROOT, ref_info.get("angles_path", "")))
    if os.path.exists(ref_angles_path):
        with open(ref_angles_path, "r") as f:
            reference_data = json.load(f)
    else:
        reference_data = {"frames": [{"frame_idx": fs["frame_idx"], "angles": fs.get("mean_angles", {})} for fs in stats_data["frames"]]}

    alignment_path, p_phases, r_phases = sync_engine.sync_videos(practice_data, reference_data)
    
    # Colored Terminal Output for Phases
    print(f"\033[92m\n--- Analysis for Shot Type: {shot_type} ---")
    print(f"Practice Phases (Frames): {p_phases}")
    print(f"Reference Phases (Frames): {r_phases}\n\033[0m")

    mapping = {p: r for p, r in alignment_path}

    # --- Scoring ---
    joint_weights = {
        "left_elbow": 1.5, "right_elbow": 1.5, 
        "left_shoulder": 1.2, "right_shoulder": 1.2, 
        "left_hip": 1.0, "right_hip": 1.0, 
        "left_knee": 0.8, "right_knee": 0.8,
        "left_wrist": 0.5, "right_wrist": 0.5,
        "left_ankle": 0.5, "right_ankle": 0.5
    }
    total_score, total_weight = 0, 0
    for p_idx, r_idx in mapping.items():
        if p_idx < len(practice_data["frames"]) and r_idx < len(stats_data["frames"]):
            p_angles = practice_data["frames"][p_idx].get("angles", {})
            stat_frame = stats_data["frames"][r_idx]
            q1_a, q3_a = stat_frame.get("q1_angles", {}), stat_frame.get("q3_angles", {})
            min_a_ref, max_a_ref = stat_frame.get("min_angles", {}), stat_frame.get("max_angles", {})
            
            for joint, weight in joint_weights.items():
                if joint in p_angles and joint in q1_a and joint in q3_a:
                    val = p_angles[joint]
                    q1, q3 = q1_a[joint], q3_a[joint]
                    mn, mx = min_a_ref.get(joint, q1-10), max_a_ref.get(joint, q3+10)
                    s = 100 if q1 <= val <= q3 else (100*(val-mn)/(q1-mn) if val<q1 and mn!=q1 else 100*(mx-val)/(mx-q3) if val>q3 and mx!=q3 else 0)
                    total_score += max(0, s) * weight
                    total_weight += weight
                    
    final_percentage = (total_score / total_weight) if total_weight > 0 else 0
    feedback_str = f"### Overall Accuracy Score: {final_percentage:.1f}%\nBiomechanical sync complete."

    # --- LLM Feedback Generation ---
    # Analyze status specifically at the Strike phase for the most impactful feedback
    joint_status_at_strike = {}
    if p_phases and 'strike' in p_phases:
        p_strike_idx = p_phases['strike']
        r_strike_idx = mapping.get(p_strike_idx)
        
        if r_strike_idx is not None and p_strike_idx < len(practice_data["frames"]) and r_strike_idx < len(stats_data["frames"]):
            p_angles = practice_data["frames"][p_strike_idx].get("angles", {})
            stat_frame = stats_data["frames"][r_strike_idx]
            q1_a, q3_a = stat_frame.get("q1_angles", {}), stat_frame.get("q3_angles", {})
            min_a_ref, max_a_ref = stat_frame.get("min_angles", {}), stat_frame.get("max_angles", {})
            
            for joint in joint_weights.keys():
                if joint in p_angles:
                    val = p_angles[joint]
                    
                    if joint in q1_a and joint in q3_a:
                        q1, q3 = q1_a[joint], q3_a[joint]
                        mn, mx = min_a_ref.get(joint, q1-10), max_a_ref.get(joint, q3+10)
                        
                        if q1 <= val <= q3:
                            status = "Ideal"
                        elif mn <= val <= mx:
                            status = "Slightly Off"
                        else:
                            status = "Needs Work"
                        
                        joint_status_at_strike[joint] = {
                            "status": status,
                            "angle": f"{val:.1f} deg",
                            "ideal_range": f"{q1:.0f}-{q3:.0f} deg"
                        }
                    else:
                        # Fallback for joints without statistical data
                        joint_status_at_strike[joint] = {
                            "status": "N/A (No Stats)",
                            "angle": f"{val:.1f} deg",
                            "ideal_range": "N/A"
                        }

    # Call LLM Engine
    print("\033[94m[LLM] Generating specialized biomechanical report with full trend analysis...\033[0m")
    llm_report_text = llm_engine.generate_feedback(
        practice_data["frames"], 
        stats_data["frames"], 
        shot_type, 
        p_phases, 
        r_phases
    )
    
    # Wrap feedback in a styled HUD container for a premium look
    html_report = llm_report_text.replace('\n', '<br>')
    styled_feedback = f"""
<div style="font-family:'Rajdhani', sans-serif; background:rgba(0, 255, 136, 0.02); border-left:4px solid #00ff88; padding:20px; border-radius:0 12px 12px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
    <h3 style="margin-top:0; color:#00ff88; text-transform:uppercase; letter-spacing:2px; font-size:14px; font-weight:700;">Biomechanical Analysis Report</h3>
    <div style="color:#94a3b8; line-height:1.6; font-size:15px;">
        {html_report}
    </div>
</div>
"""
    out_video = create_synced_video(practice_video, ref_video, alignment_path, progress_callback=progress_callback)
    return out_video, styled_feedback

def generate_interactive_widget(pose_json_path, shot_type="None"):
    import json, os, math
    import numpy as np

    if not pose_json_path or not os.path.exists(pose_json_path):
        return "<p style='color:gray'>No pose data available.</p>"

    with open(pose_json_path, "r") as f:
        frames = json.load(f)

    if not frames:
        return "<p style='color:gray'>No pose data available.</p>"

    # ---- Average landmarks and angles across all frames ----
    all_landmarks, all_angles = {}, {}
    for frame in frames:
        for k, v in frame.get("landmarks", {}).items():
            all_landmarks.setdefault(k, []).append([v["x"], v["y"]])
        for k, v in frame.get("angles", {}).items():
            all_angles.setdefault(k, []).append(v)

    avg_landmarks = {
        k: {"x": float(np.mean([p[0] for p in pts])),
            "y": float(np.mean([p[1] for p in pts]))}
        for k, pts in all_landmarks.items()
    }
    avg_angles = {k: float(np.mean(v)) for k, v in all_angles.items() if v}
    
    if not avg_landmarks:
        return "<p style='color:gray'>Pose landmarks missing from analysis.</p>"

    # ---- Geometric fallback ----
    def angle_from_pts(a, b, c):
        if not all(n in avg_landmarks for n in (a, b, c)): return None
        A, B, C = avg_landmarks[a], avg_landmarks[b], avg_landmarks[c]
        ba = (A["x"]-B["x"], A["y"]-B["y"])
        bc = (C["x"]-B["x"], C["y"]-B["y"])
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag = math.sqrt(ba[0]**2+ba[1]**2) * math.sqrt(bc[0]**2+bc[1]**2)
        return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag)))) if mag > 1e-9 else None

    TRIPLETS = {
        "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
        "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_hip": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
    }

    def resolve_angle(jname):
        for key in (f"{jname}_angle", jname, f"angle_{jname}", jname.replace("_","")):
            if key in avg_angles: return avg_angles[key]
        t = TRIPLETS.get(jname)
        return angle_from_pts(*t) if t else None

    # ---- Ideal ranges & Tips ----
    IDEAL_RANGES = {
        "default": {"left_elbow_angle":(150,180),"right_elbow_angle":(150,180),"left_knee_angle":(140,175),"right_knee_angle":(140,175),"left_hip_angle":(160,180),"right_hip_angle":(160,180),"left_shoulder_angle":(80,120),"right_shoulder_angle":(80,120)},
        "flick": {"left_elbow_angle":(80,140),"right_elbow_angle":(100,160),"left_knee_angle":(130,165),"right_knee_angle":(140,175),"left_hip_angle":(150,175),"right_hip_angle":(150,175),"left_shoulder_angle":(60,100),"right_shoulder_angle":(70,110)},
        "cover": {"left_elbow_angle":(140,175),"right_elbow_angle":(100,150),"left_knee_angle":(140,165),"right_knee_angle":(150,175),"left_hip_angle":(155,175),"right_hip_angle":(155,175),"left_shoulder_angle":(75,110),"right_shoulder_angle":(75,115)},
        "defense": {"left_elbow_angle":(150,180),"right_elbow_angle":(150,180),"left_knee_angle":(145,175),"right_knee_angle":(145,175),"left_hip_angle":(160,180),"right_hip_angle":(160,180),"left_shoulder_angle":(85,120),"right_shoulder_angle":(85,120)},
        "pull": {"left_elbow_angle":(70,130),"right_elbow_angle":(70,130),"left_knee_angle":(120,160),"right_knee_angle":(120,160),"left_hip_angle":(140,170),"right_hip_angle":(140,170),"left_shoulder_angle":(50,90),"right_shoulder_angle":(50,90)},
        "sweep": {"left_elbow_angle":(110,155),"right_elbow_angle":(110,155),"left_knee_angle":(100,145),"right_knee_angle":(110,150),"left_hip_angle":(130,165),"right_hip_angle":(130,165),"left_shoulder_angle":(60,100),"right_shoulder_angle":(60,100)},
    }
    JOINT_TIPS = {
        "default": {"left_elbow":"Keep elbows relaxed and close to the body.","right_elbow":"Drive through with the right elbow leading.","left_knee":"Bend the front knee to stay balanced.","right_knee":"Flex the back knee for a stable base.","left_hip":"Stay upright at the hips for a straight bat.","right_hip":"Rotate the back hip for power.","left_shoulder":"Point the front shoulder at the bowler.","right_shoulder":"Bring the back shoulder through the line."},
        "flick": {"left_elbow":"Bend the front elbow early to power the flick.","right_elbow":"Drive wrist rotation with the right elbow.","left_knee":"Firm front leg creates the lever.","right_knee":"Push off the back knee for weight transfer.","left_hip":"Stable hip is the pivot point.","right_hip":"Rotate through to add pace.","left_shoulder":"Lock the front shoulder.","right_shoulder":"Follow through for full extension."},
        "cover": {"left_elbow":"Extend through the line of the ball.","right_elbow":"Keep tucked to avoid an open bat face.","left_knee":"Drive off a bent front knee.","right_knee":"Back knee low transfers weight forward.","left_hip":"Lead with the front hip.","right_hip":"Let back hip open naturally.","left_shoulder":"Point front shoulder cover-wards.","right_shoulder":"End shoulder over the left foot."},
        "defense": {"left_elbow":"Keep high and close to guide ball down.","right_elbow":"Up and in - prevents bat angling away.","left_knee":"Soft bend to absorb pace.","right_knee":"Slight flex - avoid locking out.","left_hip":"Stay tall; collapsing causes edges.","right_hip":"Stay sideways; do not rotate hip.","left_shoulder":"Stay high and closed for straight bat.","right_shoulder":"Do not open shoulder early."},
        "pull": {"left_elbow":"Bend sharply to swing across the line.","right_elbow":"Drive down through the ball.","left_knee":"Stay low - bend both knees.","right_knee":"Deep flex creates coiled power.","left_hip":"Pivot early to make room.","right_hip":"Explosive rotation for power.","left_shoulder":"Drop slightly to get under short ball.","right_shoulder":"Roll over to keep ball low."},
        "sweep": {"left_elbow":"Lead low - stay close to the pad.","right_elbow":"Drop to guide bat across the line.","left_knee":"Front knee down for a good sweep.","right_knee":"Back knee low for balance.","left_hip":"Stay low; rising up causes mistiming.","right_hip":"Rotate to sweep square or fine.","left_shoulder":"Aim at the leg side target.","right_shoulder":"Full rotation completes follow-through."},
    }
    GENERAL_TIPS = {"default":"Maintain balance and smooth swing.","defense":"Keep bat close to pad, head over ball.","flick":"Use wrist rotation and firm base.","cover":"Lead with shoulder, transfer weight.","pull":"Stay low, roll wrists.","sweep":"Stay balanced, use front knee."}

    shot_key = shot_type.lower().replace(" ","_")
    if shot_key not in IDEAL_RANGES: shot_key = "default"
    ranges, joint_tips = IDEAL_RANGES[shot_key], JOINT_TIPS.get(shot_key, JOINT_TIPS["default"])
    general_tip = GENERAL_TIPS.get(shot_key, GENERAL_TIPS["default"])

    def joint_status(akey, val):
        if akey not in ranges: return "Tracked", "#58a6ff", "ref"
        lo, hi = ranges[akey]
        margin = (hi-lo)*0.3
        if lo <= val <= hi: return "Ideal", "#3fb950", "ideal"
        if lo-margin <= val <= hi+margin: return "Slightly off", "#d29922", "warn"
        return "Needs work", "#f85149", "bad"

    connections = [("left_shoulder","right_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_wrist"),("right_shoulder","right_elbow"),("right_elbow","right_wrist"),("left_shoulder","left_hip"),("right_shoulder","right_hip"),("left_hip","right_hip"),("left_hip","left_knee"),("left_knee","left_ankle"),("right_hip","right_knee"),("right_knee","right_ankle")]
    xs, ys = [v["x"] for v in avg_landmarks.values()], [v["y"] for v in avg_landmarks.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)

    def norm(vx, vy, W=328, H=428, pad=40):
        sx = (vx-minx)/(maxx-minx+1e-6)
        sy = (vy-miny)/(maxy-miny+1e-6)
        return round(pad+sx*(W-2*pad),1), round(pad+sy*(H-2*pad),1)

    TRACKED = ["left_shoulder","right_shoulder","left_elbow","right_elbow","left_hip","right_hip","left_knee","right_knee"]
    joint_data = {}
    for jname in TRACKED:
        if jname not in avg_landmarks: continue
        lm = avg_landmarks[jname]
        cx, cy = norm(lm["x"], lm["y"])
        val = resolve_angle(jname)
        akey = f"{jname}_angle"
        if val is not None:
            status, color, skey = joint_status(akey, val)
            lo_hi = ranges.get(akey)
            ideal_str = f"{lo_hi[0]}-{lo_hi[1]}" if lo_hi else ""
        else:
            status, color, skey = "Tracked", "#00e5ff", "ref"
            ideal_str = ""
        joint_data[jname] = {"cx": cx, "cy": cy, "color": color, "status": status, "skey": skey, "angle": f"{val:.1f}" if val is not None else "", "ideal": ideal_str, "tip": joint_tips.get(jname, "Focus on correct form."), "label": jname.replace("_"," ").title()}

    lines_svg = "".join([f'<line x1="{norm(avg_landmarks[a]["x"],avg_landmarks[a]["y"])[0]}" y1="{norm(avg_landmarks[a]["x"],avg_landmarks[a]["y"])[1]}" x2="{norm(avg_landmarks[b]["x"],avg_landmarks[b]["y"])[0]}" y2="{norm(avg_landmarks[b]["x"],avg_landmarks[b]["y"])[1]}" stroke="rgba(0, 229, 255, 0.2)" stroke-width="3" stroke-linecap="round"/>' for a, b in connections if a in avg_landmarks and b in avg_landmarks])
    circles_svg = "".join([f'<circle id="jc_{jn}" class="cpw_joint" cx="{jd["cx"]}" cy="{jd["cy"]}" r="8" fill="{jd["color"]}" stroke="#0a0a0c" stroke-width="2"><title>{jd["label"]}: {jd["angle"]}&deg;</title></circle>' for jn, jd in joint_data.items()])
    table_rows = "".join([f'<tr id="tr_{jn}" style="cursor:pointer;border-bottom:1px solid rgba(0, 255, 136, 0.1);"><td style="padding:8px 10px;color:#94a3b8;font-size:12px;white-space:nowrap;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{jd["color"]};margin-right:7px;vertical-align:middle;"></span>{jd["label"]}</td><td style="padding:8px 6px;color:{jd["color"]};font-weight:600;font-size:13px;">{jd["angle"]}&deg;</td><td style="padding:8px 6px;font-size:11px;color:#94a3b8;">{jd["ideal"]}&deg;</td><td style="padding:8px 6px;"><span style="font-size:10px;padding:2px 7px;border-radius:4px;background:{"rgba(63, 185, 80, 0.1)" if jd["skey"]=="ideal" else "rgba(210, 153, 34, 0.1)" if jd["skey"]=="warn" else "rgba(248, 81, 73, 0.1)" if jd["skey"]=="bad" else "rgba(0, 229, 255, 0.1)"};color:{jd["color"]};font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">{jd["status"]}</span></td></tr>' for jn, jd in joint_data.items()])

    js_data = json.dumps(joint_data)
    shot_display = shot_type if shot_type and shot_type.lower() != "none" else "General"

    html = f"""
<style>
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: #0a0a0c; }}
  ::-webkit-scrollbar-thumb {{ background: #00ff88; border-radius: 10px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #00e5ff; }}
  * {{ scrollbar-width: thin; scrollbar-color: #00ff88 #0a0a0c; }}
  .cpw_joint {{ transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer; }}
  .cpw_joint:hover {{ filter: brightness(1.2) drop-shadow(0 0 8px currentColor); transform-origin: center; }}
</style>
<div id="cpw" style="font-family:'Rajdhani',sans-serif;background:#0a0a0c;color:#ffffff;border-radius:12px;overflow:hidden;border:1px solid rgba(0, 255, 136, 0.15);box-shadow: 0 0 30px rgba(0,0,0,0.5);">
  <div style="background:#121217;border-bottom:1px solid rgba(0, 255, 136, 0.1);padding:12px 18px;display:flex;align-items:center;justify-content:space-between;">
    <div><span style="font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#00ff88;font-weight:700;">Pose Analysis</span><span style="margin-left:10px;font-size:10px;background:rgba(0, 229, 255, 0.1);color:#00e5ff;padding:2px 10px;border-radius:4px;border:1px solid rgba(0, 229, 255, 0.2);text-transform:uppercase;font-weight:600;">{shot_display}</span></div>
    <div style="font-size:11px;color:#94a3b8;font-family:'Share Tech Mono',monospace;">{len(frames)} frames analyzed</div>
  </div>
  <div style="display:flex;">
    <div style="flex:0 0 auto;background:#0a0a0c;border-right:1px solid rgba(0, 255, 136, 0.1);padding:16px;display:flex;flex-direction:column;align-items:center;">
      <svg id="cpw_svg" viewBox="0 0 328 428" width="230" height="310" style="background:#0a0a0c;">
        {lines_svg}{circles_svg}<circle id="cpw_ring" cx="-999" cy="-999" r="15" fill="none" stroke="#00ff88" stroke-width="2.5" stroke-dasharray="5 3" opacity="0"/>
      </svg>
    </div>
    <div style="flex:1;min-width:0;display:flex;flex-direction:column;background:#121217;">
      <div id="cpw_detail" style="padding:16px 18px;background:rgba(0, 255, 136, 0.03);border-bottom:1px solid rgba(0, 255, 136, 0.1);min-height:145px;display:flex;align-items:center;justify-content:center;">
        <div style="color:#94a3b8;font-size:14px;text-align:center;font-style:italic;">Select a joint for deep analysis</div>
      </div>
      <div style="overflow-y:auto;max-height:300px;"><table id="cpw_table" style="width:100%;border-collapse:collapse;font-size:12px;"><tbody>{table_rows}</tbody></table></div>
      <div style="padding:12px 18px;background:#121217;border-top:1px solid rgba(0, 255, 136, 0.1);margin-top:auto;"><p style="font-size:10px;color:#00ff88;margin:0 0 4px;text-transform:uppercase;letter-spacing:1px;font-weight:700;">Coaching Insight</p><p style="font-size:12px;color:#94a3b8;margin:0;line-height:1.4;">{general_tip}</p></div>
    </div>
  </div>
</div>
<script>
var CPW_DATA={js_data},CPW_CUR=null;
function cpwSel(n){{ if(CPW_CUR===n) return; if(CPW_CUR){{ var e=document.getElementById('jc_'+CPW_CUR); if(e){{e.setAttribute('r','8');e.setAttribute('stroke','#0a0a0c');e.setAttribute('stroke-width','2');}} var r=document.getElementById('tr_'+CPW_CUR); if(r)r.style.background=''; }} CPW_CUR=n; var jd=CPW_DATA[n], el=document.getElementById('jc_'+n); if(el){{el.setAttribute('r','13');el.setAttribute('stroke',jd.color);el.setAttribute('stroke-width','3');}} var row=document.getElementById('tr_'+n); if(row)row.style.background='rgba(0, 255, 136, 0.08)'; var ring=document.getElementById('cpw_ring'); if(ring){{ring.setAttribute('cx',jd.cx);ring.setAttribute('cy',jd.cy);ring.setAttribute('stroke',jd.color);ring.setAttribute('opacity','1');}}
var bbg=jd.skey==='ideal'?'rgba(63, 185, 80, 0.15)':jd.skey==='warn'?'rgba(210, 153, 34, 0.15)':jd.skey==='bad'?'rgba(248, 81, 73, 0.15)':'rgba(0, 229, 255, 0.15)';
document.getElementById('cpw_detail').innerHTML=`<div style="display:flex;gap:14px;width:100%;"><div style="width:48px;height:48px;border-radius:50%;background:${{bbg}};border:2px solid ${{jd.color}};display:flex;align-items:center;justify-content:center;box-shadow:0 0 15px ${{jd.color}}44;"><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="${{jd.color}}" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2"/></svg></div><div style="flex:1;"><div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><span style="font-size:16px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">${{jd.label}}</span><span style="font-size:9px;padding:2px 8px;border-radius:4px;background:${{bbg}};color:${{jd.color}};font-weight:700;text-transform:uppercase;">${{jd.status}}</span></div><div style="display:flex;gap:24px;margin-bottom:12px;"><div><div style="font-size:9px;color:#94a3b8;letter-spacing:1px;">DETECTED</div><div style="font-size:28px;font-weight:800;color:${{jd.color}};font-family:'Share Tech Mono',monospace;">${{jd.angle}}&deg;</div></div><div><div style="font-size:9px;color:#94a3b8;letter-spacing:1px;">TARGET</div><div style="font-size:28px;font-weight:800;color:#94a3b8;font-family:'Share Tech Mono',monospace;">${{jd.ideal}}&deg;</div></div></div><div style="background:rgba(0,0,0,0.2);padding:10px;border-left:3px solid ${{jd.color}};border-radius:0 4px 4px 0;"><p style="font-size:12px;color:#94a3b8;margin:0;line-height:1.4;">${{jd.tip}}</p></div></div></div>`; }}
Object.keys(CPW_DATA).forEach(n=>{{ var e=document.getElementById('jc_'+n); if(e)e.onclick=()=>cpwSel(n); var r=document.getElementById('tr_'+n); if(r)r.onclick=()=>cpwSel(n); }});
</script>
"""
    import base64
    encoded = base64.b64encode(html.encode('utf-8')).decode('utf-8')
    return f'<iframe src="data:text/html;base64,{encoded}" width="100%" height="620px" style="border:none; border-radius:12px;"></iframe>'

