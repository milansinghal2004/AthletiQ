import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
ATHLETIQ_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'AthletiQ'))
if ATHLETIQ_ROOT not in sys.path:
    sys.path.insert(0, ATHLETIQ_ROOT)

try:
    from app.main_dashboard import segment_player
except Exception as exc:
    print(json.dumps({"success": False, "error": f"Failed to import AthletiQ backend: {exc}"}))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run AthletiQ analysis from Node server')
    parser.add_argument('--video', required=True, help='Path to uploaded video file')
    parser.add_argument('--shot', default='None', help='Shot type selected by the user')
    parser.add_argument('--user_id', type=int, default=None, help='Logged in user ID')
    parser.add_argument('--clickx', type=int, default=320, help='Click X coordinate for segmentation')
    parser.add_argument('--clicky', type=int, default=240, help='Click Y coordinate for segmentation')
    args = parser.parse_args()

    try:
        result = segment_player(
            os.path.abspath(args.video),
            [args.clickx, args.clicky],
            args.shot,
            args.user_id
        )
        if not result:
            print(json.dumps({"success": False, "error": "Analysis returned no data."}))
            sys.exit(1)

        isolated_path, bio_json, comparison_path, score_feedback = result
        output = {
            "success": True,
            "video_name": os.path.basename(args.video),
            "shot_type": args.shot,
            "summary": score_feedback,
            "isolated_path": isolated_path,
            "bio_json": bio_json,
            "comparison_path": comparison_path,
            "score": score_feedback,
            "isolated_url": isolated_path and f"/outputs/{os.path.basename(isolated_path)}",
            "comparison_url": comparison_path and f"/outputs/{os.path.basename(comparison_path)}"
        }
        print(json.dumps(output))
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == '__main__':
    main()