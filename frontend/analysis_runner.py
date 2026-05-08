import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# The app directory is in the parent of the frontend folder
ATHLETIQ_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
if ATHLETIQ_ROOT not in sys.path:
    sys.path.insert(0, ATHLETIQ_ROOT)

import traceback
import contextlib

# Suppress stdout during imports to prevent non-JSON noise
with contextlib.redirect_stdout(sys.stderr):
    try:
        from app.core.pipeline import pipeline
    except Exception as exc:
        print(f"IMPORT ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
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

    # Use stderr for all logs to keep stdout clean for the JSON result
    print(f"Runner starting with video: {args.video}", file=sys.stderr)
    if not os.path.exists(args.video):
        print(f"CRITICAL: Video file not found at {args.video}", file=sys.stderr)
        print(json.dumps({"success": False, "error": "Video file not found."}))
        sys.exit(1)

    try:
        # Use the unified pipeline
        print("Invoking pipeline.process...", file=sys.stderr)
        result, error = pipeline.process(
            os.path.abspath(args.video),
            [args.clickx, args.clicky],
            args.shot
        )
        
        if error:
            print(f"Pipeline returned error: {error}", file=sys.stderr)
            print(json.dumps({"success": False, "error": error}))
            sys.exit(1)

        print("Analysis successful. Preparing output...", file=sys.stderr)
        output = {
            "success": True,
            "video_name": os.path.basename(args.video),
            "shot_type": result["shot_type"],
            "summary": result["feedback"],
            "isolated_path": result["isolated_video"],
            "bio_json": result["biomechanics_json"],
            "comparison_path": result["sync_video"],
            "score": result["feedback"],
            "isolated_url": result["isolated_video"] and f"/outputs/{os.path.basename(result['isolated_video'])}",
            "comparison_url": result["sync_video"] and f"/outputs/{os.path.basename(result['sync_video'])}"
        }
        # THIS is the only thing that should be on stdout
        print(json.dumps(output))
        sys.exit(0)
    except Exception as exc:
        print(f"EXCEPTION in main: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"success": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == '__main__':
    main()